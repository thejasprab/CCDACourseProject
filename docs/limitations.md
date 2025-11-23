# Limitations

This system demonstrates a complete Spark-based recommender pipeline, but several structural limitations remain due to the nature of arXiv metadata and the constraints of scalable, single-machine execution.

---

## 1. Dataset-Level Limitations

### 1.1 Metadata-only text (no full paper content)
The system relies solely on titles, abstracts, and categories because the dataset does not include PDF full text.  
This limits semantic depth and restricts the recommender to high-level lexical similarity rather than full scientific understanding.

### 1.2 Inconsistent or incomplete metadata
Some records contain missing authors, categories, or malformed abstracts (LaTeX artifacts), which can distort category trends, co-occurrence graphs, and author-based analytics.

---

## 2. Pipeline & System Limitations

### 2.1 Single-machine resource constraints
Training TF-IDF and running complex SQL analyses on the full arXiv dataset requires significant memory and disk spill space.  
These limitations reflect hardware constraints rather than Spark’s capability on distributed clusters.

### 2.2 Streaming is a controlled simulation
The streaming workflow processes synthetic weekly batches rather than live external data, which limits realism but satisfies the course requirement to demonstrate Structured Streaming.
