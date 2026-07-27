# NELSR: Nepali Legal Statute Retrieval Benchmark

NELSR is the first comprehensive benchmark for **Nepali case-to-statute retrieval**, designed to evaluate automated retrieval systems that identify relevant statutory provisions for legal cases.

Legal professionals often spend significant time searching through lengthy and hierarchically structured legal documents to identify applicable statutes. While legal retrieval systems have advanced significantly for high-resource languages, the Nepali legal domain lacks a standardized benchmark for realistic case-to-statute retrieval and systematic evaluation.

NELSR addresses this gap by introducing:

* **Case-derived retrieval queries** extracted from Supreme Court judgments
* **Federal statutes** as the retrieval corpus
* A **citation-aware query construction** approach for reliable case-to-statute relevance pairs
* **Legal structure-aware segmentation** to preserve statutory hierarchy

The benchmark evaluates sparse, dense, late-interaction, and hybrid retrieval approaches using standard information retrieval metrics.

---


# Repository Structure

```text
NELSR/
├── data/
│   ├── chunks/                  # Structure-aware statute chunks
│   ├── queries/                 # Case-derived retrieval queries
│   ├── valid_names.json         # Legal document names
│   └── stopwords_ne.json        # Nepali stopword list
│
├── chunk_generation/
│   └── build_chunk.py           # Scripts used for chunk generation 
│
├── query_generation/
│   ├── build_query.py           # Scripts used for query generation 
│   └── clean_query.py           # Query cleaning and filtering
│
├── experiments/
│   └── experiment.ipynb         # Dense and hybrid retrieval experiments
│
├── bm25.ipynb                   # BM25 baseline experiment
│
└── README.md
```

Note: The `chunk_generation` and `query_generation` directories contain the code used to construct the NELSR benchmark. The original legal documents and Supreme Court case documents used during data collection are not redistributed due to source restrictions. To reproduce the benchmark construction process, users need to obtain the original documents from their respective public sources and run the provided scripts. The released processed chunks and queries are sufficient to reproduce the retrieval experiments.


# Dataset

The released NELSR benchmark consists of:

* **Queries:** Case-derived retrieval queries paired with relevant statutory provisions
* **Chunks:** Structure-aware statute segments used as retrieval units

The original legal documents were collected from publicly available legal sources but are not redistributed. This repository provides the processed benchmark resources required for retrieval experiments.

## Dataset Structure

The expected directory structure is:

```text
data/
├── chunks/
│   └── chunks.jsonl              #Generated statute chunks used for retrieval experiments
│
├── queries/
│   ├── queries.jsonl              # Generated retrieval queries
│   └── cleaned_queries.jsonl      # Cleaned queries with citations removed
│
├── valid_names.json               # Legal document names used during query extraction
│
└── stopwords_ne.json              # Nepali stopword list used during preprocessing
```

### Chunk Format

Each chunk contains:

* `text`: Text content of the statute segment used for retrieval
* `article_heading`: Legal article heading associated with the chunk
* `sub_article_id`: Identifier of the referenced sub-article

The `text` field is used for generating embeddings during dense retrieval experiments, while `sub_article_id` is used for relevance evaluation.

### Query Format

Each query contains:

* `query_text`: Retrieval query constructed from judicial decisions
* `relevant_subarticles`: Ground-truth statutory provisions relevant to the query

---

# Installation

NELSR uses **Poetry** for dependency management.

Install dependencies:

```bash
poetry install
```

---

# Running Experiments

## 1. BM25 Baseline

The BM25 sparse retrieval experiment can be reproduced using:

```text
bm25.ipynb
```

Run all cells sequentially to evaluate the BM25 baseline on NELSR.

---

## 2. Dense and Hybrid Retrieval

The dense and hybrid retrieval experiments require GPU resources and were conducted using Google Colab.

Run:

```text
experiments/experiment.ipynb
```

Execute all cells sequentially to reproduce:

* Dense retrieval experiments
* Hybrid retrieval experiments
* Retrieval evaluation

GPU acceleration is recommended for dense and hybrid retrieval experiments.

---


# Evaluation Metrics

NELSR evaluates retrieval performance using:

* Recall@K
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (nDCG)

The benchmark evaluates retrieval effectiveness for Nepali case-to-statute retrieval using sparse, dense, late-interaction, and hybrid retrieval approaches.

---

# Results

The best-performing hybrid retrieval approach achieves:

| Metric     | Score |
| ---------- | ----: |
| Recall@100 |  0.66 |
| nDCG@10    |  0.34 |
| MRR@10     |  0.32 |
