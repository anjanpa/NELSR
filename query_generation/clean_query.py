import argparse
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class QueryCleaner:

    def __init__(self, query_path):
        self.query_path = query_path
        self.df = None
        self.filtered_df = None

    def load_queries(self):
        self.df = pd.read_json(
            self.query_path,
            lines=True
        )

        self.df["query_text"] = (
            self.df["query_text"]
            .fillna("")
        )

        # Keep original query
        self.df["original_query_text"] = self.df["query_text"]

        # Hide citations for cleaning
        self.df["query_text"] = self.df.apply(
            lambda row: self.hide_citations(
                row["query_text"],
                row.get("raw_citation", "")
            ),
            axis=1
        )

        return self.df


    def actual_words(self, text):
        words = []

        for token in text.split():
            token = token.strip(".,;:!?()[]{}\"'")

            if "।" in token:
                continue

            if re.search(r"[0-9०-९]", token):
                continue

            if re.fullmatch(r"[ऀ-ॿ]{3,}", token):
                words.append(token)

        return words

    def hide_citations(self, text, citation):
        """
        Remove the raw citation from query text.
        """

        if not citation:
            return text

        return text.replace(citation, "")

    
    def split_sentences(self, text):

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

        sentences = []
        start = 0

        for match in sentence_pattern.finditer(text):

            end = match.end()
            part = text[start:end]

            if any(
                re.search(pattern, part[-30:])
                for pattern in ignored_patterns
            ):
                continue

            if part.strip():
                sentences.append(part.strip())

            start = end

        if start < len(text):
            sentences.append(text[start:].strip())

        return sentences


    def get_context_word_counts(self, row):

        text = row["query_text"]
        citation = row.get("raw_citation", "")

        sentences = self.split_sentences(text)

        citation_idx = None

        for i, sentence in enumerate(sentences):
            if citation and citation in sentence:
                citation_idx = i
                break

        # If citation not found, use full query length
        if citation_idx is None:
            return [
                0,
                len(self.actual_words(text)),
                0
            ]

        before = sum(
            len(self.actual_words(s))
            for s in sentences[:citation_idx]
        )

        citation_sentence = len(
            self.actual_words(
                sentences[citation_idx]
            )
        )

        after = sum(
            len(self.actual_words(s))
            for s in sentences[citation_idx+1:]
        )

        return [
            before,
            citation_sentence,
            after
        ]


    def remove_duplicates(self, threshold=1.0):

        texts = self.df["query_text"].tolist()

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2),
            min_df=2
        )

        matrix = vectorizer.fit_transform(texts)

        similarity = cosine_similarity(matrix)

        remove_ids = set()

        n = len(self.df)

        for i in range(n):

            for j in range(i+1, n):

                if similarity[i][j] >= threshold:
                    remove_ids.add(j)


        self.df = self.df.drop(
            index=list(remove_ids)
        ).reset_index(drop=True)


        print(
            "Duplicates removed:",
            len(remove_ids)
        )

        return self.df


    def remove_length_outliers(self):

        self.df["sentence_lengths"] = self.df.apply(
            self.get_context_word_counts,
            axis=1
        )


        lengths = self.df[
            "sentence_lengths"
        ].apply(lambda x: x[1])


        Q1 = lengths.quantile(0.25)
        Q3 = lengths.quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR


        print("Length bounds:", lower, upper)


        before = len(self.df)

        self.df = self.df[
            lengths.between(lower, upper)
        ].copy()


        print(
            "Length outliers removed:",
            before - len(self.df)
        )

        return self.df


    def clean(self):

        self.load_queries()

        print(
            "Original queries:",
            len(self.df)
        )

        self.remove_duplicates()

        self.remove_length_outliers()

        self.filtered_df = self.df.reset_index(
            drop=True
        )

        print(
            "Final queries:",
            len(self.filtered_df)
        )

        return self.filtered_df


    def save_jsonl(self, output_path):

        self.filtered_df.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False
        )




def main():

    parser = argparse.ArgumentParser(
        description="Clean legal queries by removing duplicates and length outliers"
    )

    parser.add_argument(
        "--query_path",
        default="data/queries/queries.jsonl",
        help="Path to input JSONL query file"
    )

    parser.add_argument(
        "--output_path",
        default="data/queries/cleaned_queries.jsonl",
        help="Path to save cleaned JSONL file"
    )


    args = parser.parse_args()


    cleaner = QueryCleaner(
        args.query_path
    )

    filtered_df = cleaner.clean()

    cleaner.save_jsonl(
        args.output_path
    )

    print(
        f"Saved cleaned queries to {args.output_path}"
    )


if __name__ == "__main__":
    main()