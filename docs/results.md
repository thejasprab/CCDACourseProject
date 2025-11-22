
# Results

## 1. Ingestion Results

### ✔ Full Dataset
- Raw Input: `arxiv-metadata-oai-snapshot.json` (Kaggle)
- Processed Output: `data/processed/arxiv_full/`
- Partitioning: by **year**
- Compression: **ZSTD**
- Total Rows: Automatically logged during ingestion

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
- Vocabulary size: 250,000  
- min_df: 5  
- Extra stopwords: enabled (extra_stopwords_topdf=500)
- Output paths:
  - `data/models/tfidf_sample/`
  - `data/processed/features_sample/`

### Full Model
- Vocabulary size: 120,000  
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
- Cosine similarity (`topk_exact`)
- Query vectorization (`vectorize_query`)

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
- Works for both sample + full modes via:
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

Generated outputs per batch:
- `by_year.csv`
- `papers_per_year.png`
- `doi_rate_by_year.csv`
- `doi_rate_by_year.png`
- top_categories.png

### Full Streaming
A weekly full-snapshot streaming simulator:
```
reports/streaming_full/YYYYMMDD/
```

Outputs are identical in structure but operate on the full dataset.

These streaming jobs use:
- `transform_all` on micro‑batches
- Repartitioning for performance
- Spark Structured Streaming (trigger: once or micro‑batch)

---

## 5. Complex SQL Analytics Results

Running:
```
python -m pipelines.complex_full
```

Generates a full set of **10 advanced analytics**:

### 1. Category Co‑Occurrence
- CSV: `complex_category_cooccurrence.csv`
- Plot: `complex_category_cooccurrence_top.png`

### 2. Author Collaboration Over Time
- Collaboration pairs by year
- Totals and top pairs (plots)

### 3. Rising / Declining Topics
- Percent changes from earliest→latest year
- Top 20 rising + declining topics

### 4. Readability / Lexical Trends
- Lexical richness by year
- Abstract length trends

### 5. DOI vs Versions Correlation
- Pearson correlation
- Group-level comparison

### 6. Author Productivity Lifecycle
- Scatter: active years vs paper count

### 7. Author Category Migration
- Cross‑category transitions
- Top 20 migration paths

### 8. Abstract Length vs Popularity
- Correlation
- Decile analysis

### 9. Weekday Submission Patterns
- Submissions by weekday

### 10. Category Stability via Versions
- Average versions per category

Outputs stored under:
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
- Full TF‑IDF model
- Complex analytics CSVs + PNGs
- (Optional) streaming update batch

All outputs are reproducible and stored in:
```
data/processed/
data/models/
reports/analysis_full/
reports/streaming_full/
```

---

## 7. Summary

The Sparxiv project successfully:
- Processes a full arXiv dataset
- Normalizes + enriches metadata
- Computes feature embeddings
- Performs exact cosine similarity search
- Executes 10 sophisticated Spark SQL analytics
- Supports both full streaming + sample streaming modes
- Generates all required CSVs, metrics, and visualizations
