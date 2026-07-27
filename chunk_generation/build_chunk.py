import argparse
import json
import re
import os
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent


DOCUMENT_SOURCES = {
    "law": BASE_DIR / "data/legal_documents/laws" / "docx_files",
    "law_converted": BASE_DIR / "data/legal_documents/laws" / "doc_converted_files",
    "constitution": BASE_DIR / "data/legal_documents/constitution" / "docx_files",
}

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
SPACE_RE = re.compile(r"\s+")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

PART_RE = re.compile(r"^(भाग|परिच्छेद)\s*[-–—]?\s*([०-९0-9]+)?\s*$")
SCHEDULE_RE = re.compile(r"^अनुसूची\s*[-–—]?\s*[०-९0-9]*")
BODY_START_RE = re.compile(
    r"(प्रस्तावना|यो ऐन बनाई|ऐन बनाई|ऐन बनाएको छ|जारी गरिबक्सेको छ)"
)
ARTICLE_HEADING_RE = re.compile(
    r"^\s*([०-९0-9]+)\s*[.)।]\s*(.+?)(?:\s*[:ः]\s*(.*)|\s*)$"
)
SUBCLAUSE_RE = re.compile(r"^\s*\([०-९0-9क-हa-zA-Z]+\)")

# SUBARTICLE_MARKER_RE = re.compile(r"(?<!\S)\(([०-९0-9क-हa-zA-Z]+)\)")
SUBARTICLE_MARKER_RE = re.compile(
    r"(?:^|\s{2,})\(([०-९0-9]+|[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह]+)\)"
)

def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\xa0", " ")
    value = ZERO_WIDTH_RE.sub("", value)
    value = SPACE_RE.sub(" ", value)
    return value.strip()

def normalize_number(value: str) -> str:
    return normalize_text(value).translate(DEVANAGARI_DIGITS)

def read_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as docx:
        document_xml = docx.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t"))
        text = normalize_text(text)
        if text:
            paragraphs.append(text)
    return paragraphs

def read_doc_paragraphs(path: Path) -> list[str]:
    """
    Best-effort support for old .doc files when a converter exists locally.
    """
    commands = [
        ["antiword", str(path)],
        ["catdoc", str(path)],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
            continue

        return [
            text
            for text in (normalize_text(line) for line in result.stdout.splitlines())
            if text
        ]

    return []


def read_document_paragraphs(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx_paragraphs(path)
    if suffix == ".doc":
        return read_doc_paragraphs(path)
    return []


def infer_article_type(path: Path, title: str) -> str:
    path_text = str(path)
    if "constitution" in path_text or "संविधान" in title:
        return "धारा"
    return "दफा"

def is_article_heading(text: str) -> re.Match[str] | None:
    match = ARTICLE_HEADING_RE.match(text)
    if not match:
        return None

    title = normalize_text(match.group(2))
    if not title or SUBCLAUSE_RE.match(text):
        return None

    return match

    
def subarticle_level(label: str) -> str:
    normalized = normalize_number(label)
    if normalized.isdigit():
        return "subsection"
    return "clause"

def clean_nepali_legal_text(text):
    # 1. Remove long ellipsis/dots (..........)
    text = re.sub(r"\.{2,}", " ", text)

    # 2. Normalize danda spacing (important for sentence splitting)
    text = re.sub(r"\s*।\s*", " । ", text)

    # 3. Remove only unwanted quotes (keep brackets for numbering)
    text = re.sub(r"[\"“”‘’']", "", text)

    # 4. Normalize spacing inside brackets (optional cleanup)
    # ( १ ) → (१)
    text = re.sub(r"\(\s*([०-९]+)\s*\)", r"(\1)", text)

    # 5. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # 6. Trim stray leading punctuation left by a marker split (", …" / "– …").
    text = re.sub(r"^[\s,;।–—-]+", "", text)

    # 7. Strip
    return text.strip()


# At least one Devanagari letter/digit. Used to decide whether a chunk has real
# word content once its markers are removed.
_CONTENT_RE = re.compile(r"[०-९0-9क-ह]")
# Closing paren optional so a truncated marker like "(५" is stripped too.
_MARKER_ONLY_RE = re.compile(r"\([०-९0-9क-ह]+\)?")


def has_content(text: str) -> bool:
    """True when `text` carries real words, not just markers/punctuation.

    A body that is only a marker like ``(ठ)``/``(१)`` or bare punctuation
    (``–``, ``,–``) is treated as empty and should not become a chunk."""
    if not text:
        return False
    without_markers = _MARKER_ONLY_RE.sub(" ", text)
    return bool(_CONTENT_RE.search(without_markers))


SUBARTICLE_MARKER_RE = re.compile(
    r"(?:^|\s{2,})\(([०-९0-9]+|[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह]+)\)"
)

# An article at or below this word count is always kept whole (one chunk); a
# larger article is split. When splitting produces no markers, this is also the
# block size for the plain word-based fallback split.
ARTICLE_SPLIT_THRESHOLD = 200

# A marker-based subchunk ((१)/(क)…) is left intact up to this size; only when it
# exceeds this is it further word-split into balanced blocks of this size.
SUBCHUNK_SPLIT_THRESHOLD = 200


def marker_level(label: str) -> str:
    return "subsection" if normalize_number(label).isdigit() else "clause"


# Devanagari consonant ordering used for clause markers (क)(ख)(ग)… — index in
# this string gives a clause its sequence position.
CLAUSE_CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"


def marker_seq_value(level: str, label: str) -> int | None:
    """Sequence position of a marker within its level (1-based).

    (१)->1 (२)->2 … for numeric subsections; (क)->1 (ख)->2 … for clauses.
    Returns None when the label can't be placed in a sequence."""
    if level == "subsection":
        number = normalize_number(label)
        return int(number) if number.isdigit() else None
    head = label[:1]
    index = CLAUSE_CONSONANTS.find(head)
    return index + 1 if index >= 0 else None


DANDA = "।"


def split_sentences(text: str) -> list[str]:
    """Split on the Devanagari danda (।), keeping the danda attached to its sentence."""
    sentences: list[str] = []
    buffer = ""
    for token in re.split(r"(।)", text):
        buffer += token
        if token == DANDA:
            stripped = buffer.strip()
            if stripped:
                sentences.append(stripped)
            buffer = ""
    tail = buffer.strip()
    if tail:
        sentences.append(tail)
    return sentences


def word_blocks(text: str, size: int) -> list[str]:
    """Balanced word-count split — even blocks, no tiny tail (201 -> 101+100)."""
    words = text.split()
    n = len(words)
    pieces = (n + size - 1) // size          # ceil(n / size) -> block count
    per = (n + pieces - 1) // pieces          # ceil(n / pieces) -> balanced size
    return [" ".join(words[i:i + per]) for i in range(0, n, per)]


def split_into_blocks(text: str, size: int) -> list[str]:
    """
    Split `text` into balanced blocks of at most ~`size` words, breaking on
    sentence (danda ।) boundaries so no block starts or ends mid-sentence.

    Whole sentences are packed toward a balanced target (so there are no tiny
    tail blocks); a single sentence longer than `size` is word-split as a last
    resort so no block exceeds the cap.
    """
    total_words = len(text.split())
    if total_words <= size:
        return [text] if text.strip() else []

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return word_blocks(text, size)

    pieces = (total_words + size - 1) // size
    target = (total_words + pieces - 1) // pieces   # balanced words per block

    blocks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            blocks.append(" ".join(current))
            current = []
            current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if sentence_words > size:
            # A lone oversized sentence can't honour the cap on a boundary.
            flush()
            blocks.extend(word_blocks(sentence, size))
            continue
        if current and current_words + sentence_words > target:
            flush()
        current.append(sentence)
        current_words += sentence_words

    flush()
    return blocks


def split_oversized_subchunks(
    subarticles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Word-split any marker-based subchunk whose body exceeds the threshold.

    Blocks that are only a marker/punctuation (e.g. a lone ``(ठ)`` tail) are
    dropped and the survivors renumbered, so no content-less chunk is emitted."""
    expanded: list[dict[str, Any]] = []
    for sub in subarticles:
        blocks = [
            block
            for block in split_into_blocks(
                clean_nepali_legal_text(sub["text"]), SUBCHUNK_SPLIT_THRESHOLD
            )
            if has_content(block)
        ]
        if not blocks:
            continue
        if len(blocks) == 1:
            expanded.append({**sub, "text": blocks[0]})
            continue
        for index, block in enumerate(blocks, start=1):
            expanded.append({
                **sub,
                "text": block,
                "block_index": index,
                "block_count": len(blocks),
            })
    return expanded


def prepend_intro(
    intro_text: str, subarticles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prepend the article's pre-marker intro (chapeau) to each marker subchunk
    so every split carries its lead-in context. A whole (unsplit) article keeps
    the intro inline already, so this only applies when the article is split."""
    intro_text = clean_nepali_legal_text(intro_text)
    if not intro_text:
        return subarticles
    return [
        {**sub, "text": f"{intro_text} {clean_nepali_legal_text(sub['text'])}".strip()}
        for sub in subarticles
    ]


def _collect_markers(paragraphs: list[str]) -> list[dict[str, Any]]:
    """Ordered list of every subarticle marker across all paragraphs."""
    markers: list[dict[str, Any]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        for match in SUBARTICLE_MARKER_RE.finditer(paragraph):
            label = match.group(1)
            markers.append({
                "paragraph_index": paragraph_index,
                "start": match.start(),
                "end": match.end(),
                "level": marker_level(label),
                "label": label,
                "marker": match.group(0).strip(),
            })
    return markers


def _flag_top_level_markers(markers: list[dict[str, Any]]) -> None:
    """Mark which markers begin a *top-level* subarticle (``is_split``).

    Walks the markers as an outline, keeping a stack of open list levels. A
    marker that restarts at (१)/(क) opens a nested list (pushes a frame); a
    marker that continues an ancestor's sequence returns to that level (pops).
    Only markers resolved at depth 0 split the article, so numeric lists nested
    inside clauses — which each restart at (१) — stay inside their parent."""
    stack: list[dict[str, Any]] = []  # each: {"level", "value"}
    for marker in markers:
        marker["is_split"] = False
        level = marker["level"]
        value = marker_seq_value(level, marker["label"])
        if value is None:
            continue

        if not stack:
            stack.append({"level": level, "value": value})
            marker["is_split"] = True
            continue

        top = stack[-1]
        if top["level"] == level and value > top["value"]:
            # Advances the current level's sequence — a sibling at this depth.
            # Uses > (not == prev+1) so a skipped marker (OCR gap) still reads as
            # a sibling instead of being mis-nested.
            top["value"] = value
            marker["is_split"] = len(stack) == 1
            continue

        if value == 1:
            # Restarts a sequence -> opens a nested list one level deeper.
            stack.append({"level": level, "value": value})
            marker["is_split"] = len(stack) == 1
            continue

        # Neither a sibling of the current level nor a fresh nested list: try to
        # ascend to an ancestor whose sequence this marker continues.
        while len(stack) > 1:
            stack.pop()
            top = stack[-1]
            if top["level"] == level and value > top["value"]:
                top["value"] = value
                marker["is_split"] = len(stack) == 1
                break
        else:
            # Couldn't reconcile against any open level; keep it inline.
            top = stack[-1]
            if top["level"] == level and value > top["value"]:
                top["value"] = value
                marker["is_split"] = len(stack) == 1


def extract_subarticles_by_level(paragraphs: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Split an article at its outermost marker level only.

    Nested markers (a numeric list inside a clause, a clause inside a
    subsection) are kept inline in their parent's text rather than pulling the
    split deeper."""
    markers = _collect_markers(paragraphs)
    _flag_top_level_markers(markers)
    by_paragraph: dict[int, list[dict[str, Any]]] = {}
    for marker in markers:
        by_paragraph.setdefault(marker["paragraph_index"], []).append(marker)

    intro_parts: list[str] = []
    subarticles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def close_current() -> None:
        nonlocal current
        if current is None:
            return
        
        # A marker with no body of its own (nothing but the chapeau would remain
        # after prepend_intro) is a spurious/empty subarticle -> drop it.
        text = normalize_text(" ".join(current.pop("_parts")))
        if has_content(text):
            current["text"] = text
            subarticles.append(current)
        current = None

    def append_text(text: str) -> None:
        text = normalize_text(text)
        if not text:
            return
        if current is None:
            intro_parts.append(text)
        else:
            current["_parts"].append(text)

    for paragraph_index, paragraph in enumerate(paragraphs):
        pos = 0
        for marker in by_paragraph.get(paragraph_index, []):
            append_text(paragraph[pos:marker["start"]])
            if marker["is_split"]:
                close_current()
                current = {
                    "label": marker["label"],
                    "level": marker["level"],
                    "marker": marker["marker"],
                    "paragraph_index": paragraph_index,
                    "_parts": [],
                }
            else:
                # Nested marker -> keep its text inline in the parent subarticle.
                append_text(marker["marker"])
            pos = marker["end"]
        append_text(paragraph[pos:])

    close_current()
    return normalize_text(" ".join(intro_parts)), subarticles


def extract_subarticles(paragraphs: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """
    Chunk an article according to its size:

      - <= ARTICLE_SPLIT_THRESHOLD words -> keep the whole article as one chunk
        (a complete article is always kept whole, no matter how small).
      - otherwise split it, preferring, in order:
          1. numeric subarticles (१)(२)(३)
          2. clause markers (क)(ख)(ग)
          3. a plain ARTICLE_SPLIT_THRESHOLD-word block split.
    """
    full_text = normalize_text(" ".join(paragraphs))
    cleaned_full = clean_nepali_legal_text(full_text)
    article_word_count = len(cleaned_full.split())

    # A complete article at or below the threshold is always kept whole.
    if article_word_count <= ARTICLE_SPLIT_THRESHOLD:
        return full_text, []

    # Split at the OUTERMOST marker level only. This keeps a definitions-style
    # article (क)(ख)…(ङ) split at the clause level even though (ङ) nests numeric
    # (१)(२) inside it, and keeps an article whose subsections (१)(२) each nest
    # clause lists (क)(ख)… split at the subsection level — in both directions the
    # nested markers stay inside their parent instead of pulling the split deeper.
    intro, subs = extract_subarticles_by_level(paragraphs) 
    if subs:
        return intro, split_oversized_subchunks(prepend_intro(intro, subs))

    # No markers at all -> fall back to a plain word-size split.
    subarticles = [
        {
            "label": None,
            "level": "wordsplit",
            "marker": None,
            "paragraph_index": None,
            "text": chunk,
        }
        for chunk in split_into_blocks(cleaned_full, ARTICLE_SPLIT_THRESHOLD)
    ]
    return "", subarticles


def extract_articles_from_paragraphs(
    paragraphs: list[str],
    source_path: Path,
    document_group: str,
) -> dict[str, Any]:
    title = paragraphs[0] if paragraphs else source_path.stem
    article_type = infer_article_type(source_path, title)

    articles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_part: str | None = None
    current_chapter: str | None = None
    body_started = False

    def close_current() -> None:
        nonlocal current
        if current is None:
            return
        article_paragraphs = current.pop("_paragraphs")
        
        current["text"] = normalize_text(" ".join(article_paragraphs))
        intro_text, subarticles = extract_subarticles(article_paragraphs)
        current["intro_text"] = intro_text
        current["subarticles"] = subarticles
        articles.append(current)
        current = None
    
    for index, paragraph in enumerate(paragraphs):
        if SCHEDULE_RE.match(paragraph):
            close_current()
            break

        if BODY_START_RE.search(paragraph):
            body_started = True

        part_match = PART_RE.match(paragraph)
        if part_match:
            body_started = True
            current_part = paragraph if part_match.group(1) == "भाग" else current_part
            current_chapter = paragraph if part_match.group(1) == "परिच्छेद" else current_chapter
            continue

        heading_match = is_article_heading(paragraph)
        if heading_match and not body_started:
            continue

        if heading_match:
            close_current()

            number_raw = heading_match.group(1)
            heading = normalize_text(heading_match.group(2))
            rest = normalize_text(heading_match.group(3))
            body = rest if rest else paragraph[heading_match.end(2):].lstrip(" :ः")

            current = {
                "document_title": title,
                "document_group": document_group,
                "source_file": str(source_path),
                "article_type": article_type,
                "article_number": number_raw,
                "article_number_normalized":normalize_number(number_raw),
                "heading": heading,
                "part": current_part,
                "chapter": current_chapter,
                "_paragraphs": [body] if body else [],
            }
            continue

        if current is not None:
            current["_paragraphs"].append(paragraph)
    close_current()

    # Compute last article number
    last_article_number = None

    if articles:
        try:
            last_article_number = max(
                int(a["article_number_normalized"]) for a in articles
            )
        except ValueError:
            last_article_number = None
            
    return {
        "document_title": title,
        "document_group": document_group,
        "source_file": str(source_path),
        "article_type": article_type,
        "article_count": len(articles),
        "last_article_number": last_article_number,
        "is_sequential": (
            last_article_number == len(articles)
            if last_article_number is not None else None
        ),
        "articles": articles,
    }


def iter_source_files(
    sources: dict[str, Path]
) -> list[tuple[Path, str]]:
    files = []

    for group, directory in sources.items():
        for path in sorted(directory.glob("*.doc*")):
            files.append((path, group))

    return files



def build_structured_documents(
    limit: int | None = None,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    subarticles_f = open(BASE_DIR/"data/chunks/subarticles.jsonl", "w", encoding="utf-8")

    document_id=1
    article_id=1
    sub_article_id=1
    low_article=0

    for index, (path, group) in enumerate(iter_source_files(DOCUMENT_SOURCES)):
        if limit is not None and index >= limit:
            break

        paragraphs = read_document_paragraphs(path)
        if not paragraphs:
            skipped.append({
                "source_file": str(path),
                "reason": "Could not read document text",
            })
            continue

        structured = extract_articles_from_paragraphs(paragraphs, path, group)
        for article in structured["articles"]:
            article["text"]=clean_nepali_legal_text(article["text"])
            article["intro_text"]=clean_nepali_legal_text(article.get("intro_text",""))
            heading=clean_nepali_legal_text(article["heading"])
            if len(article["text"].split())<1 or not heading:
                low_article+=1
                continue
            
            article["document_id"]= document_id
            article["article_id"]=article_id 

            # write subarticles
            subarticles = article.get("subarticles", [])

            if subarticles:
                for sub in subarticles:
                    sub["text"] = clean_nepali_legal_text(sub.get("text", ""))
                    if has_content(sub["text"]):
                        sub_record = {
                            "document_id": document_id,
                            "article_id": article_id,
                            "sub_article_id": sub_article_id,
                            "document_title": structured["document_title"],
                            "article_number": article["article_number"],
                            "article_type": article["article_type"],
                            "article_heading":article["heading"],
                            **sub,
                        }
                        subarticles_f.write(json.dumps(sub_record, ensure_ascii=False) + "\n")
                        sub_article_id += 1
            else:
                # Article has no subarticles -> store article itself as a subarticle
                sub_record = {
                    "document_id": document_id,
                    "article_id": article_id,
                    "sub_article_id": sub_article_id,
                    "document_title": structured["document_title"],
                    "article_number": article["article_number"],
                    "article_type": article["article_type"],
                    "article_heading":article["heading"],
                    "level": None,
                    "label": None,
                    "marker":None,
                    "text": article["text"], 
                }
                subarticles_f.write(json.dumps(sub_record, ensure_ascii=False) + "\n")
                sub_article_id += 1
            article_id+=1 

        if not structured["articles"]:
            skipped.append({
                "source_file": str(path),
                "reason": "No numbered article headings found",
            })
            continue

        documents.append(structured)
        print(f"{path.name}: {structured['article_count']}-{structured['last_article_number']} {structured['article_type']} ")
        document_id+=1 

    subarticles_f.close()

    return {
        "summary": {
            "documents_processed": len(documents),
            "documents_skipped": len(skipped),
            "total_articles": sum(doc["article_count"] for doc in documents),
        },
        "next_document_id": document_id,
        "next_article_id": article_id,
        "next_sub_article_id": sub_article_id,
        "documents": documents,
        "skipped": skipped,
    }

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_structured_documents(
        limit=args.limit,
    )

    summary = result["summary"]
    print("\nDone.")
    print(f"  Documents : {summary['documents_processed']}")
    print(f"  Articles  : {summary['total_articles']}")
    print(f"  Skipped   : {summary['documents_skipped']}")

if __name__ == "__main__":
    main()
