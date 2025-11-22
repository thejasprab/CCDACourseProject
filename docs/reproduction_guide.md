# Reproduction Guide

This guide provides step-by-step instructions to fully reproduce the Sparxiv pipeline:
from raw arXiv metadata → ingestion → TF‑IDF training → search → analytics.  

---

# 1. Requirements

## 1.1 System Requirements
- Python 3.9+
- Java 11+
- Apache Spark 3.x (handled automatically by PySpark)
- At least **16GB RAM** recommended for FULL pipeline
- **60–200GB free disk space** for full arXiv snapshot + processing

## 1.2 Python Dependencies  
Install via:

```
pip install -r requirements.txt
```

Main packages:
- pyspark
- pandas
- numpy
- matplotlib
- requests (for Kaggle downloader)
- pyarrow (for Parquet support)

---

# 2. Directory Structure

The pipeline assumes:

```
data/
  raw/
    arxiv-metadata-oai-snapshot.json  (downloaded automatically)
  processed/
  models/
reports/
```

If missing, they will be created automatically.

---

# 3. Step-by-Step Reproduction

---

# 3.1 Download the Dataset

You can either download manually from Kaggle:

- https://www.kaggle.com/datasets/Cornell-University/arxiv

or let the project download automatically using:

```
python -m streaming.kaggle_downloader --mode full
```

This creates:

```
data/raw/arxiv-metadata-oai-snapshot.json
```

---

# 3.2 Run Full Pipeline (Recommended)

The repository includes a complete end-to-end script:

```
bash run.sh
```

This performs:

1. Download dataset (only if missing)
2. Ingestion → Parquet  
3. TF‑IDF model training  
4. Complex analytics  
5. (Optional) Streaming updates  

Artifacts will appear under:

```
data/processed/arxiv_full/
data/models/tfidf_full/
data/processed/features_full/
reports/analysis_full/
```

---

# 3.3 Running Individual Stages

---

## 3.3.1 Ingest Only

### FULL dataset:
```
python -m pipelines.ingest_full
```

### SAMPLE dataset:
```
python -m pipelines.ingest_sample
```

Both run:

```
run_ingestion(
    input_path=...,
    output_path=...,
    partition_by="year",
    min_abstract_len=40
)
```

---

## 3.3.2 Train TF-IDF Model Only

### Full model:
```
python -m pipelines.train_full
```

### Sample model:
```
python -m pipelines.train_sample
```

Outputs:

```
data/models/tfidf_full/
data/processed/features_full/
```

---

## 3.3.3 Run Search

You may launch the Flask app in `app/server.py`:

```
python app/server.py
```

Or test from Python:

```python
from engine.search.search_engine import SearchEngine

engine = SearchEngine(mode="sample")
results = engine.search(
    title="graph neural networks",
    abstract="message passing architecture",
    k=5
)
print(results)
```

---

## 3.3.4 Run Complex Analytics Only

```
python -m pipelines.complex_full
```

This generates:
- CSV summaries  
- PNG charts  
for all 10 analytical modules.

---

# 4. Streaming Reproduction

To simulate incoming daily arXiv updates:

```
python -m streaming.sample_stream
```

Drop files into:

```
data/stream/incoming_sample/
```

Reports will appear under:

```
reports/streaming_sample/YYYYMMDD/
```

---

# 5. Validating the Output

## 5.1 Validate Ingestion
Check schema:

```
spark.read.parquet("data/processed/arxiv_full").printSchema()
```

Should include:
- primary_category  
- year  
- authors_list  
- category_list  
- abstract_len  
- has_doi  

## 5.2 Validate Model
```
ls data/models/tfidf_full/
```

Expect:
- `stages/`
- `model.json`

## 5.3 Validate Features
```
spark.read.parquet("data/processed/features_full").show()
```

Columns:
- id_base  
- paper_id  
- title  
- abstract  
- categories  
- year  
- features (SparseVector)

---

# 6. Troubleshooting

### RAM issues during full training:
- Increase driver memory:
```
export SPARK_DRIVER_MEMORY=8g
```

### OOM during ingestion:
- Reduce shuffle partitions:
```
export SPARK_SQL_SHUFFLE_PARTITIONS=200
```

### Slow ingestion:
- Avoid multiline JSON unless necessary
- Use local SSD storage

### Dataset missing:
Ensure:
```
data/raw/arxiv-metadata-oai-snapshot.json
```
exists—or re-run run.sh.

---

# 7. Summary

This reproduction guide provides every step necessary to:

1. Acquire the arXiv dataset  
2. Ingest + clean using Spark  
3. Train TF-IDF models  
4. Compute analytical reports  
5. Perform similarity search  
6. (Optionally) process streaming updates  

Running `run.sh` fully reproduces the original results.

