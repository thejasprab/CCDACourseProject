# Results 

## 1. Ingestion Results

### ✔ Full Dataset
- Raw Input: `arxiv-metadata-oai-snapshot.json` (Kaggle)
- Processed Output: `data/processed/arxiv_full/`
- Partitioning: by **year**
- Compression: **ZSTD**
- Total Rows: Logged automatically during ingestion

### ✔ Sample Dataset
- Input: `data/sample/arxiv-sample.jsonl`
- Output: `data/processed/arxiv_sample/`

### Key Stats Produced
- **Top 10 primary categories**
- **Paper counts by year**
- **Average lengths of titles & abstracts**
- **Author and category counts**

These appear in the run logs during `pipelines.ingest_full` and `pipelines.ingest_sample`.

---

## 2. TF‑IDF Feature Model Results

### Sample Model
- Vocabulary size: **80,000**
- min_df: 3
- Extra stopwords: enabled (extra_stopwords_topdf=500)
- Output paths:
  - `data/models/tfidf_sample/`
  - `data/processed/features_sample/`

### Full Model
- Vocabulary size: **250,000**
- min_df: 10
- Extra stopwords: disabled (performance reasons)
- Output paths:
  - `data/models/tfidf_full/`
  - `data/processed/features_full/`

### Generated Columns
Each feature parquet contains:
- `id_base`
- `paper_id`
- `title`
- `abstract`
- `categories`
- `year`
- `features` (L2‑normalized SparseVector)

These constitute the foundation for both **search** and **complex analytics**.

---

## 3. Search Engine Results

The search system uses:
- Spark ML pipeline (TF-IDF)
- Exact cosine similarity (`topk_exact`)
- Query vectorization (`vectorize_query`)
- **Full‑mode search uses an offline‑built CSR matrix to avoid Spark cross‑joins, enabling scalable exact similarity search over millions of papers.**

### Example Output (structure)
```json
{
  "rank": 1,
  "score": 0.873,
  "neighbor_id": "1234.5678",
  "paper_id": "1234.5678v1",
  "title": "Deep Learning Approaches for...",
  "abstract": "We propose a novel...",
  "categories": ["cs.LG"],
  "year": 2022
}
```

### Demo Behavior
- Top‑K (default k=10)
- Search using title only, abstract only, or both
- Supports both sample + full modes:
```python
from engine.search.search_engine import SearchEngine
SearchEngine(mode="full").search(...)
```

---

## 4. Streaming Results

The project includes **two streaming pipelines**:

### Sample Streaming
Outputs under:
```
reports/streaming_sample/YYYYMMDD/
```

Generated per microbatch:
- `by_year.csv`
- `papers_per_year.png`
- `doi_rate_by_year.csv`
- `doi_rate_by_year.png`
- `top_categories.png`

### Full Streaming
Simulated weekly full snapshots:
```
reports/streaming_full/YYYYMMDD/
```

Outputs mirror the sample pipeline but operate on the full dataset.

Streaming jobs use:
- `transform_all` during micro‑batches
- Repartitioning for performance
- Spark Structured Streaming (trigger: once or micro‑batch)

---

## 5. Complex SQL Analytics Results

Running:
```
python -m pipelines.complex_full
```

Produces **10 advanced analytics**:

1. Category Co‑Occurrence  
2. Author Collaboration Over Time  
3. Rising / Declining Topics  
4. Lexical Richness & Abstract Length Trends  
5. DOI vs Versions Correlation  
6. Author Productivity Lifecycle  
7. Author Category Migration  
8. Abstract Length vs Popularity  
9. Weekday Submission Patterns  
10. Category Stability (versions)

All outputs stored under:
```
reports/analysis_full/
```

---

## 6. End‑to‑End Pipeline Results

Running:
```
bash run.sh
```

Produces:
- Full ingestion → parquet
- TF‑IDF model (sample + full)
- CSR index (full mode)
- Complex analytics CSVs + PNGs
- Optional streaming updates

Stored under:
```
data/processed/
data/models/
reports/analysis_full/
reports/streaming_full/
```

---

## 7. Summary

The Sparxiv pipeline successfully:
- Processes the complete arXiv dataset
- Generates TF‑IDF features
- Uses CSR for scalable full‑dataset similarity search
- Executes 10 major analytical workflows
- Supports reproducible ingestion, analytics, search, and streaming
