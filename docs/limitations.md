# Limitations

Sparxiv is intentionally built as a realistic but compact Spark based recommender and analytics system. That also means it inherits several important limitations from both the dataset and the modeling choices.

This document summarizes those limitations so they are explicit to anyone reusing or extending the code.

---

## 1. Dataset Level Limitations

### 1.1 Metadata only, no full text

The Kaggle arXiv snapshot only includes metadata and abstracts, not full paper PDFs or source.

- Title and abstract are strong signals but do not capture all technical nuance.
- References, formulae, figures, and detailed proofs are not modeled at all.
- The recommender is biased toward topics that are easy to summarize lexically and may miss deeper relationships that only appear in the body of the paper.

Extending the system to use the public PDF buckets is possible but would require a significantly more complex text extraction pipeline.

### 1.2 Noisy and inconsistent fields

- Abstracts often contain LaTeX commands, math delimiters, and inline markup.
- Author strings can be inconsistent across versions and categories.
- Comments, report numbers, and journal references follow no single schema.

The ingestion step performs minimal cleaning for robustness and speed, which means some noise is carried into the TF-IDF vocabulary and analytics.

### 1.3 Temporal and versioning ambiguity

arXiv allows multiple versions for the same paper id. The dataset does not always clearly distinguish:

- The exact timeline of category changes across versions.
- Submission versus revision events for authors and categories.
- Differences between versions beyond simple counts.

Analyses that rely on time, such as category migration, author productivity over time, and rising or declining topics, are therefore approximate and can be skewed by how version metadata is encoded.

### 1.4 Category imbalance and bias

The category distribution is highly skewed:

- A small set of areas dominate submission counts.
- Niche areas have very small sample sizes.

This introduces:

- Bias in TF-IDF vocabulary construction, since dominant categories drive IDF.
- Unstable statistics for rare categories in complex analytics.
- Potential over emphasis on trends in already popular domains.

---

## 2. Modeling Limitations

### 2.1 Lexical TF-IDF is non semantic

The recommender uses TF-IDF with cosine similarity:

- It does not model synonyms or paraphrases.
- It does not understand cross domain concepts unless the vocabulary overlaps strongly.
- It ignores word order except for optional bigrams, which are disabled in the provided configs.
- It cannot represent deeper semantics, only patterns in token co occurrence.

### 2.2 Exact top K search is linear

Both sample and full modes use exact similarity search:

- In sample mode, the query vector is compared against every document vector in memory.
- In full mode, the CSR matrix multiplication still scales linearly in the number of documents.

There is no approximate nearest neighbor index. While the CSR based approach is efficient enough for a single user demo, it would not scale to heavy traffic or significantly larger corpora without additional engineering.

---

## 3. Pipeline and System Limitations

### 3.1 Single machine constraints

Although Spark is capable of running on clusters, this project is configured for a single machine setup:

- Memory and partition settings are tuned for local execution.
- Large shuffles can still spill heavily to disk and be slow.
- The full pipeline on the complete arXiv snapshot can approach the limits of commodity hardware.

Scaling to a multi node cluster would require retuning Spark shuffle and partitioning parameters and may require changes to how intermediate files are handled.

### 3.2 Index rebuilds are offline and heavy

The full CSR index:

- Is built offline by scanning the entire features Parquet.
- Must be rebuilt from scratch after major changes to the underlying features.
- Must be fully loaded in memory by the web app at startup.

There is no incremental index update path, and index construction can be substantial in runtime and memory use on very large corpora.

---

## 4. Summary

In short:

- The data is rich but incomplete, since it is based on metadata and abstracts.
- The model is scalable and transparent but lexical and non semantic.
- The system is reproducible and reasonably fast on a single machine but not production hardened.

Keep these in mind if you use Sparxiv as a base for further research or real world applications.
