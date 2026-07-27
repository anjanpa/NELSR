import argparse
import json
import re
import time
import unicodedata
from bisect import bisect_right
from pathlib import Path


# Where files live by default
BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent/"data"

DEFAULT_NAMES_PATH       = DATASET_DIR / "valid_names.json"
DEFAULT_DETAILS_DIR      = DATASET_DIR / "case_documents"
DEFAULT_OUTPUT_PATH      = DATASET_DIR  / "queries" / "queries.jsonl"
DEFAULT_SUBARTICLES_PATH = DATASET_DIR / "chunks" / "subarticles.jsonl"

DEFAULT_WORD_WINDOW     = 100
DEFAULT_SENTENCE_WINDOW = 1


# Cleaning up text
class TextCleaner:
    """Cleans up messy text: invisible characters, odd spacing, and so on."""

    hidden_characters = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
    extra_spaces = re.compile(r"\s+")

    @classmethod
    def clean(cls, text):
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\xa0", " ")
        text = cls.hidden_characters.sub("", text)
        text = cls.extra_spaces.sub(" ", text)
        return text.strip()


# The list of official act names, and matching a citation against it
class NameList:
    """Holds the official act names and checks whether a citation names one of them exactly."""

    def __init__(self, names):
        self.names = names  # already cleaned, unique, longest names first
        self.lookup_table = {}
        for name in self.names:
            self.lookup_table[TextCleaner.clean(name)] = name

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        cleaned_names = []
        for item in raw_list:
            cleaned = TextCleaner.clean(str(item))
            if cleaned:
                cleaned_names.append(cleaned)

        unique_names = list(set(cleaned_names))
        unique_names.sort(key=len, reverse=True)  # check longer names before shorter ones
        return NameList(unique_names)

    def find_exact_match(self, act_name_raw):
        act_name = TextCleaner.clean(act_name_raw).rstrip()
        if act_name in self.lookup_table:
            return self.lookup_table[act_name], 1.0
        return None, 0.0


# Reading a single court document and pulling out its body text
class DocumentReader:
    """Reads one court document file and extracts the clean body of the order/judgment."""

    leading_number = re.compile(r'^\s*[\(\[]?[०-९0-9]+[\)\]]?\s*[।.)]\s*')

    body_start_words = {"आदेश", "फैसला"}
    body_end_phrase = "उक्त रायमा सहमत छु ।"

    
    def nepali_to_english_year(self,year):
        nepali_digits = "०१२३४५६७८९"
        english_digits = "0123456789"

        table = str.maketrans(nepali_digits, english_digits)
        return int(year.translate(table))

    def __init__(self, file_path):
        self.file_path = file_path
        self.paragraphs = self._load_paragraphs()
        self.edition=self._load_edition() 

    def _load_edition(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        edition=data.get("edition",None)
        if edition:
            return self.nepali_to_english_year(edition["साल"])

        return None

    def _load_paragraphs(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_paragraphs = data.get("paragraphs", [])
        if not isinstance(raw_paragraphs, list):
            return []

        cleaned_paragraphs = []
        for p in raw_paragraphs:
            cleaned = TextCleaner.clean(p)
            if cleaned:
                cleaned_paragraphs.append(cleaned)
        return cleaned_paragraphs

    def _strip_leading_number(self, paragraph):
        paragraph = self.leading_number.sub("", paragraph)  # drop things like "५।" or "(5)"
        return paragraph.replace("\n", " ").strip()

    def get_body_text(self):
        """Finds the part of the document between the order/judgment heading and the
        judges' concurrence line, and returns it as one cleaned block of text."""
        paragraphs = self.paragraphs
        total = len(paragraphs)

        start = None
        for i, p in enumerate(paragraphs):
            if p.strip() in self.body_start_words:
                start = i
                break
        if start is None:
            start = min(20, total)

        end = None
        for i in range(start + 1, total):
            if self.body_end_phrase in paragraphs[i]:
                end = i
                break
        if end is None:
            end = max(start, total - 10)

        body_paragraphs = paragraphs[start + 2:end]
        cleaned_paragraphs = [self._strip_leading_number(p) for p in body_paragraphs if p.strip()]
        return TextCleaner.clean(" ".join(cleaned_paragraphs))


# Building a window of surrounding words around a citation
class TextWindowMaker:
    """Given where a citation sits in the text, works out a chunk of surrounding words
    (some words before it and some words after it) to use as a query."""

    def __init__(self, words_before_and_after=80):
        self.window_size = words_before_and_after

    def _split_into_words(self, text):
        words = []
        word_start_positions = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            while i < n and not text[i].isspace():
                i += 1
            words.append(text[start:i])
            word_start_positions.append(start)
        return words, word_start_positions

    def get_window(self, text, citation_start, citation_end):
        words, word_start_positions = self._split_into_words(text)
        if not words:
            return None

        first_word_index = max(0, bisect_right(word_start_positions, citation_start) - 1)
        last_word_index = max(0, bisect_right(word_start_positions, citation_end) - 1)

        window_start_index = first_word_index - self.window_size
        window_end_index = last_word_index + self.window_size

        if window_start_index < 0 or window_end_index >= len(words):
            return None  # not enough words before or after to build a full window

        window_start = word_start_positions[window_start_index]
        if window_end_index == len(words) - 1:
            window_end = len(text)
        else:
            window_end = word_start_positions[window_end_index + 1]

        return window_start, window_end


# Building a window of surrounding sentences around a citation
class SentenceWindowMaker:
    """Extracts sentences before and after the citation sentence."""

    sentence_pattern = re.compile(
        r"।(?=\s+[\u0900-\u097F])|"
        r"\?(?=\s+[\u0900-\u097F])",
        re.UNICODE
    )

    ignored_patterns = [
        r"ने\s*।\s*का\s*।\s*प\s*।",
        r"न\s*।\s*नं\s*।",
        r"अ\s*।\s*बं\s*।",
        r"गा\s*।\s*वि\s*।\s*स\s*।",
        r"जि\s*।",
        r"कि\s*।",
    ]

    def __init__(self, sentences_before=2, sentences_after=2):
        self.sentences_before = sentences_before
        self.sentences_after = sentences_after

    def _split_into_sentences(self, text):
        spans = []
        start = 0

        for match in self.sentence_pattern.finditer(text):

            # text before this danda
            part = text[start:match.end()]

            # skip abbreviation cases
            if any(re.search(p, part[-30:]) for p in self.ignored_patterns):
                continue

            if part.strip():
                spans.append((start, match.end()))

            start = match.end()

        if start < len(text) and text[start:].strip():
            spans.append((start, len(text)))

        return spans
    
        
    def get_window(self, text, citation_start, citation_end):
        sentence_spans = self._split_into_sentences(text)

        if not sentence_spans:
            return None

        citation_sentence_index = None

        for i, (start, end) in enumerate(sentence_spans):
            if start <= citation_start <= end:
                citation_sentence_index = i
                break

        if citation_sentence_index is None:
            return None

        # Check enough sentences exist before citation
        if citation_sentence_index < self.sentences_before:
            return None

        # Check enough sentences exist after citation
        if citation_sentence_index + self.sentences_after >= len(sentence_spans):
            return None

        window_start_index = citation_sentence_index - self.sentences_before
        window_end_index = citation_sentence_index + self.sentences_after

        return (
            sentence_spans[window_start_index][0],
            sentence_spans[window_end_index][1]
        )

# Pulling apart a section reference, e.g. "धारा १२ को उपदफा (२)"
class SectionReader:
    """Breaks a citation's section text down into its type, number, and any subsection/clause."""

    section_pattern = re.compile(r"(?:को)?\s*(धारा|दफा)\s*([०-९0-9]+)(.*)")
    bracket_number_pattern = re.compile(r"\(([०-९0-9a-zA-Zक-ह]+)\)")
    labelled_part_pattern = re.compile(r"(उपदफा|उपधारा|खण्ड)\s*\(([०-९0-9a-zA-Zक-ह]+)\)")

    @classmethod
    def parse(cls, section_text):
        section_text = TextCleaner.clean(section_text)
        match = cls.section_pattern.search(section_text)
        if not match:
            return {"type": None, "number": None, "subsections": [], "structured": {}, "raw": section_text}

        section_type = match.group(1)
        section_number = match.group(2)
        rest_of_text = match.group(3)

        bracket_numbers = cls.bracket_number_pattern.findall(rest_of_text)
        labelled_parts = cls.labelled_part_pattern.findall(rest_of_text)

        structured = {}
        for label, value in labelled_parts:
            if label in ["उपदफा", "उपधारा"]:
                structured["subsection"] = value
            elif label == "खण्ड":
                structured["clause"] = value

        if not structured and bracket_numbers:
            structured["subsection"] = bracket_numbers[0]
            if len(bracket_numbers) >= 2:
                structured["clause"] = bracket_numbers[1]

        return {
            "type": section_type,
            "number": section_number,
            "subsections": bracket_numbers,
            "structured": structured,
            "raw": section_text,
        }


# Looking up which sub-articles a citation points to, and scoring text overlap
class SubArticleLookup:
    """Holds every sub-article grouped by the document it belongs to, and can find
    which sub-articles a given citation is pointing to."""

    def __init__(self, file_path):
        self.by_document = self._load(file_path)

    def _load(self, file_path):
        grouped = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                grouped.setdefault(entry["document_title"], []).append(entry)
        return grouped

    def get_candidates_for(self, document_name):
        return self.by_document.get(document_name or "", [])

    def find_matches(self, query):
        """Finds the sub-articles that a citation refers to.

        Returns a pair: (matching sub-articles, is_ambiguous).

        is_ambiguous is True when the cited subsection label matches more than one
        genuinely different subsection within the article. That happens when the
        source data was split up incorrectly and the same subsection label ends up
        attached to two different pieces of content. In that case we cannot tell
        which one the citation means, so the caller should drop the query.

        A subsection that was simply broken into several word-sized pieces still
        counts as ONE match, so that case is not treated as ambiguous.
        """
        candidates = self.get_candidates_for(query.get("document_name_canonical"))

        cited_section_number = query.get("section_parsed", {}).get("number")
        cited_subsections = query.get("section_parsed", {}).get("subsections", [])
        cited_subsection = cited_subsections[0] if cited_subsections else None

        matches = []
        for a in candidates:
            if a.get("article_number") != cited_section_number:
                continue
            if cited_subsection and a.get("label") != cited_subsection:
                continue
            matches.append(a)

        is_ambiguous = False
        if cited_subsection:
            distinct_count = 0
            for a in matches:
                if not a.get("block_index") or a.get("block_index") == 1:
                    distinct_count += 1
            is_ambiguous = distinct_count > 1

        return matches, is_ambiguous

    @staticmethod
    def get_text(sub_article):
        """Combines a sub-article's heading, marker, and body into one text block,
        so it can be compared against a query for word overlap."""
        parts = [
            sub_article.get("article_heading"),
            sub_article.get("marker"),
            f": {sub_article.get('text', '')}",
        ]
        return " ".join(p for p in parts if p)


class OverlapScorer:
    """Measures how much two pieces of text have in common, word for word."""

    word_pattern = re.compile(r"\w+", re.UNICODE)

    @classmethod
    def _words_in(cls, text):
        return set(cls.word_pattern.findall(text.lower()))

    @classmethod
    def score(cls, query_text, other_text):
        """Fraction of the query's words that also appear in the other text."""
        query_words = cls._words_in(query_text)
        if not query_words:
            return 0.0
        other_words = cls._words_in(other_text)
        return round(len(query_words & other_words) / len(query_words), 4)

    @classmethod
    def difficulty_label(cls, overlap_score, threshold=0.3):
        return "easy" if overlap_score >= threshold else "hard"


# Finding citations inside a block of text
class Citation:
    """One citation found in the text, e.g. 'मुलुकी अपराध संहिता, २०७४ को धारा १२'."""

    def __init__(self, start, end, act_name, section_text):
        self.start = start
        self.end = end
        self.act_name = act_name
        self.section_text = section_text

    @property
    def full_text(self):
        return self.act_name + self.section_text


class CitationFinder:
    """Scans a block of text and finds every citation in it.

    Works in two passes, which is much cheaper than one giant combined pattern:
      1. Find every place that looks like a section reference (e.g. 'धारा १२').
      2. For each one found, look backwards for an act name that ends right before it.
    """

    def __init__(self, act_names):
        self.section_pattern = re.compile(
            r"(?:को\s+)?(?:धारा|दफा)\s*[०-९0-9]+"
            r"(?:\s*\([०-९0-9क-हa-zA-Z]+\))*"
            r"(?:\s*को\s*(?:उपदफा|उपधारा|खण्ड)\s*\([०-९0-9क-हa-zA-Z]+\))*"
            r"(?:\s*,\s*\([०-९0-9क-हa-zA-Z]+\))*",
            re.UNICODE,
        )

        name_options = []
        for name in act_names:
            name_options.append(re.escape(name) + r"\s*")

        self.act_name_pattern = re.compile(r"(?:" + "|".join(name_options) + r")\Z", re.UNICODE)
        self.max_lookback_distance = max(len(n) for n in act_names) + 40

    def find_all(self, text):
        """Yields citations left to right, never letting two citations overlap."""
        last_match_end = 0
        for section_match in self.section_pattern.finditer(text):
            section_start = section_match.start()
            section_end = section_match.end()

            if section_start < last_match_end:
                continue  # this spot is already inside a citation we found

            lookback_start = max(last_match_end, section_start - self.max_lookback_distance)
            name_match = self.act_name_pattern.search(text, lookback_start, section_start)
            if name_match is None:
                continue  # no act name sits right before this section

            yield Citation(
                name_match.start(),
                section_end,
                text[name_match.start():section_start],
                text[section_start:section_end],
            )
            last_match_end = section_end


# Turning citations into query records
class QueryBuilder:
    """Turns every citation found in a document into a query record."""

    def __init__(self, citation_finder, name_list, window_maker, hide_citation=False, span_mode="word"):
        self.citation_finder = citation_finder
        self.name_list = name_list
        self.window_maker = window_maker
        self.hide_citation = hide_citation
        self.span_mode = span_mode  # "word" or "sentence" -- how query_span/query_text were built

    def build_queries(self, full_text, document_id,edition):
        queries = []
        for citation in self.citation_finder.find_all(full_text):
            citation_start, citation_end = citation.start, citation.end

            act_name_raw = TextCleaner.clean(citation.act_name)
            section_raw = TextCleaner.clean(citation.section_text)
            canonical_name, match_score = self.name_list.find_exact_match(act_name_raw)

            if canonical_name=="नेपालको संविधान" and edition<2072:
                print("skipped old constitution")
                continue 

            #creating query with word span
            window = self.window_maker.get_window(full_text, citation_start, citation_end)
            if window is None:
                continue
            window_start, window_end = window

            window_text = full_text[window_start:window_end]
            if self.hide_citation and window_start <= citation_start <= citation_end <= window_end:
                cut_start = citation_start - window_start
                cut_end = citation_end - window_start
                window_text = window_text[:cut_start] + " " + window_text[cut_end:]
            query_text = TextCleaner.clean(window_text)

            queries.append({
                "query_id":                f"{document_id}__{len(queries)}",
                "raw_citation":            TextCleaner.clean(citation.full_text),
                "document_name":           act_name_raw,
                "document_name_canonical": canonical_name,
                "matched":                 canonical_name is not None,
                "match_score":             round(match_score, 4),
                "section":                 section_raw,
                "section_parsed":          SectionReader.parse(section_raw),
                "query_text":              query_text,
                "citation_hidden":         self.hide_citation,
                "span_mode":               self.span_mode,
                "citation_span":           [citation_start, citation_end],
                "query_span":              [window_start, window_end],
            })
        return queries


# Running the whole pipeline over every document
class DocumentProcessor:
    """Reads every document in a folder, finds citations in each one, and writes
    matching queries out to a file."""

    def __init__(self, details_dir, name_list, subarticle_lookup, word_window=100,
                 sentence_before=DEFAULT_SENTENCE_WINDOW,sentence_after=DEFAULT_SENTENCE_WINDOW, span_mode="word", hide_citation=False):
        self.details_dir = details_dir
        self.name_list = name_list
        self.subarticle_lookup = subarticle_lookup

        if span_mode not in ("word", "sentence"):
            raise ValueError(f"span_mode must be 'word' or 'sentence', got {span_mode!r}")

        self.span_mode = span_mode
        self.citation_finder = CitationFinder(name_list.names)

        if span_mode == "sentence":
            self.window_maker = SentenceWindowMaker(sentence_before,sentence_after)
        else:
            self.window_maker = TextWindowMaker(word_window)

        self.query_builder = QueryBuilder(
            self.citation_finder, name_list, self.window_maker, hide_citation, span_mode
        )

        self.stats = {}

    def _score_one_query(self, query, seen_query_texts):
        """Adds overlap scores to a matched query and marks it as seen.
        Returns True if the query should be written out, False if it should be skipped."""
        query_text = query["query_text"].strip()
        if query_text in seen_query_texts:
            return False, "dup"

        matches, is_ambiguous = self.subarticle_lookup.find_matches(query)
        if is_ambiguous:
            return False, "ambiguous"
        if not matches:
            return False, "no_match"

        sub_article_ids = [m["sub_article_id"] for m in matches]
        article_ids = sorted(set(m["article_id"] for m in matches))
        query["relevant_subarticles"] = sub_article_ids
        query["relevant_articles"] = article_ids

        if len(sub_article_ids) > 4:
            return False, "too_many"

        query["sa_overlap"] = [
            OverlapScorer.score(query_text, SubArticleLookup.get_text(m)) for m in matches
        ]

        candidates = self.subarticle_lookup.get_candidates_for(query.get("document_name_canonical"))
        text_by_article = {}
        for c in candidates:
            text_by_article.setdefault(c["article_id"], []).append(SubArticleLookup.get_text(c))

        query["a_overlap"] = [
            OverlapScorer.score(query_text, " ".join(text_by_article.get(article_id, [])))
            for article_id in article_ids
        ]

        seen_query_texts.add(query_text)
        return True, "written"

    def run(self, output_path=None, limit=None, query_limit=None):
        seen_query_texts = set()

        docs_processed = docs_skipped = 0
        total_queries = total_matched = total_written = 0
        skipped_dup = skipped_ambiguous = unmatched = 0

        start_time = time.perf_counter()
        last_log_time = start_time

        output_file = None 
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = open(output_path, "w", encoding="utf-8")
            print(f"Writing queries to {output_path}")

        file_paths = sorted(self.details_dir.glob("*.json"))

        for i, file_path in enumerate(file_paths):
            print("Processing file:",file_path.name)
            # if not file_path.name =="9628.json":
            #     continue 

            if limit is not None and i >= limit:
                break
            if query_limit is not None and total_written >= query_limit:
                break
            
            document = DocumentReader(file_path)
            if len(document.paragraphs) < 10:
                docs_skipped += 1
                continue
            
            full_text = document.get_body_text()
            queries = self.query_builder.build_queries(full_text, file_path.stem,document.edition)
            
            if output_file:
                for query in queries:
                    if not query["matched"]:
                        unmatched += 1
                        continue
                    if query_limit is not None and total_written >= query_limit:
                        continue

                    should_write, reason = self._score_one_query(query, seen_query_texts)
                    if reason == "dup":
                        skipped_dup += 1
                    elif reason == "ambiguous":
                        skipped_ambiguous += 1

                    if should_write:
                        output_file.write(json.dumps(query, ensure_ascii=False) + "\n")
                        total_written += 1

            total_queries += len(queries)
            total_matched += sum(1 for q in queries if q["matched"])
            docs_processed += 1

            if docs_processed % 200 == 0:
                now = time.perf_counter()
                print(
                    f" total itered={i} [{docs_processed} docs] {now - last_log_time:.2f}s since last 200 "
                    f"({now - start_time:.2f}s total, {total_written} queries written)"
                )
                last_log_time = now

        if output_file:
            output_file.close()

        runtime = time.perf_counter() - start_time
        print(
            f"\nTOTAL: {runtime:.2f}s over {docs_processed} docs | written {total_written} | "
            f"total queries {total_queries} | unmatched {unmatched} | dup {skipped_dup} | ambiguous {skipped_ambiguous}"
        )

        self.stats = {
            "documents_processed": docs_processed,
            "documents_skipped":   docs_skipped,
            "total_queries":       total_queries,
            "total_matched":       total_matched,
            "total_unmatched":     total_queries - total_matched,
            "total_written":       total_written,
            "skipped_dup":         skipped_dup,
            "skipped_ambiguous":   skipped_ambiguous,
            "total_runtime_sec":   round(runtime, 4),
        }
        return self.stats


# Command line entry point
class CommandLineTool:
    """Reads command-line options and runs the pipeline."""

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Fast legal-citation query builder.")
        parser.add_argument("--names",           type=Path, default=DEFAULT_NAMES_PATH)
        parser.add_argument("--details-dir",     type=Path, default=DEFAULT_DETAILS_DIR)
        parser.add_argument("--output",          type=Path, default=DEFAULT_OUTPUT_PATH)
        parser.add_argument("--span-mode",       type=str,  default="sentence", choices=["word", "sentence"],
                             help="Build the query context window using word spans or sentence spans (default: word)")
        parser.add_argument("--word-window",     type=int,  default=DEFAULT_WORD_WINDOW, metavar="N",
                             help="Words before/after the citation to include when --span-mode=word")
        parser.add_argument("--sentence-before", type=int,  default=DEFAULT_SENTENCE_WINDOW, metavar="N",
                             help="Sentences before/after the citation to include when --span-mode=sentence")
        parser.add_argument("--sentence-after", type=int,  default=DEFAULT_SENTENCE_WINDOW, metavar="N",
                             help="Sentences before/after the citation to include when --span-mode=sentence")
        parser.add_argument("--limit",           type=int,  default=None, metavar="N")
        parser.add_argument("--query-limit",     type=int,  default=None, metavar="N")
        parser.add_argument("--hide-citation",   action="store_true")
        return parser.parse_args()

    @classmethod
    def run(cls):
        args = cls.parse_arguments()

        print(f"Loading {args.names}")
        name_list = NameList.load_from_file(args.names)
        print(f"  {len(name_list.names)} names loaded")

        subarticle_lookup = SubArticleLookup(DEFAULT_SUBARTICLES_PATH)

        if args.span_mode == "sentence":
            print(f"Span mode: sentence (sentence_window={args.sentence_before}--{args.sentence_after})")
        else:
            print(f"Span mode: word (word_window={args.word_window})")

        processor = DocumentProcessor(
            details_dir=args.details_dir,
            name_list=name_list,
            subarticle_lookup=subarticle_lookup,
            word_window=args.word_window,
            sentence_before=args.sentence_before,
            sentence_after=args.sentence_after,
            span_mode=args.span_mode,
            hide_citation=args.hide_citation,
        )

        stats = processor.run(
            output_path=args.output,
            limit=args.limit,
            query_limit=args.query_limit,
        )

        print(
            f"Done. {stats['documents_processed']} docs, {stats['documents_skipped']} skipped"
            f" {stats['total_queries']} queries ({stats['total_written']} written) -> {args.output}"
        )


if __name__ == "__main__":
    CommandLineTool.run()