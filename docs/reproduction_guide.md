# Reproduction Guide

This guide describes how to fully reproduce the Sparxiv pipeline:

Raw arXiv metadata to ingestion to TF-IDF training to full index to search to analytics to streaming.

It is written assuming a single machine setup with enough memory to handle the full metadata snapshot.

---

## 1. Requirements

### 1.1 System requirements

- Python 3.9 or newer
- Java 17 or newer
- Local install of PySpark 3.5.x via the `pyspark` package
- Recommended RAM:
  - 8 GB for sample only
  - 16 GB or more for the full pipeline
- Disk space:
  - 60 - 200 GB free for raw snapshot, processed Parquet, models, and temporary shuffle files

### 1.2 Python dependencies

Install from the project root:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

- pyspark
- pandas
- pyarrow
- matplotlib
- jupyter
- kagglehub
- Flask
- scipy

### 1.3 Environment variables (optional)

Tune Spark memory via environment variables:

```bash
export SPARK_DRIVER_MEMORY=10g
export SPARK_EXECUTOR_MEMORY=10g
```

Spark will also use `data/tmp/spark-local` for local spill files. Ensure this path has enough free space.

---

## 2. Directory Layout

The project assumes the following structure, with most directories created automatically:

```text
sparxiv/
  data/
    raw/
      arxiv-metadata-oai-snapshot.json
    sample/
      arxiv-sample.jsonl
    processed/
      arxiv_full/
      arxiv_sample/
      features_full/
      features_sample/
      full_index/
    stream/
      incoming/
      incoming_sample/
      checkpoints_*/
  reports/
    analysis_full/
    analysis_sample/
    standard_queries_full/
    standard_queries_sample/
    streaming_full/YYYYMMDD/
    streaming_sample/YYYYMMDD/
  app/
  engine/
  pipelines/
  streaming/
```

---

## 3. Getting the Dataset

### 3.1 Automatic download

From the project root:

```bash
python -m streaming.kaggle_downloader --mode full
```

This uses KaggleHub to download the latest `arxiv-metadata-oai-snapshot.json` and places it at:

```text
data/raw/arxiv-metadata-oai-snapshot.json
```

For the sample pipeline:

```bash
python -m streaming.kaggle_downloader --mode sample --sample-size 50000
```

This creates:

```text
data/sample/arxiv-sample.jsonl
```

### 3.2 Manual download

Alternatively, download the dataset from Kaggle and copy the JSONL file to:

```text
data/raw/arxiv-metadata-oai-snapshot.json
```

---

## 4. One Shot Pipelines

### 4.1 Full pipeline

From the project root:

```bash
bash run.sh
```

This script:

1. Ensures the full raw dataset exists, using `streaming.kaggle_downloader` if needed.
2. Ingests full JSONL to Parquet:

   ```bash
   python -m pipelines.ingest_full
   ```

   - Input: `data/raw/arxiv-metadata-oai-snapshot.json`
   - Output: `data/processed/arxiv_full/`

3. Trains the full TF-IDF model:

   ```bash
   python -m pipelines.train_full
   ```

   - Output:
     - `data/models/tfidf_full/`
     - `data/processed/features_full/`

4. Runs complex analytics on the full dataset:

   ```bash
   python -m pipelines.complex_full
   ```

   - Output: `reports/analysis_full/`

5. Optionally runs streaming on the full dataset if the corresponding line is uncommented in `run.sh`.

6. Builds the full CSR search index:

   ```bash
   python -m pipelines.build_full_index
   ```

   - Output: `data/processed/full_index/` with:
     - `full_index_csr.npz`
     - `full_index_ids.npy`
     - `full_index_paper_ids.npy`
     - `full_index_titles.npy`
     - `full_index_abstracts.npy`
     - `full_index_categories.npy`
     - `full_index_years.npy`

### 4.2 Sample pipeline

For a faster end to end demo:

```bash
bash run_sample.sh
```

This script:

1. Ensures `data/sample/arxiv-sample.jsonl` exists.
2. Runs `pipelines.ingest_sample` to create `data/processed/arxiv_sample/`.
3. Trains the sample TF-IDF model:

   ```bash
   python -m pipelines.train_sample
   ```

   - Output:
     - `data/models/tfidf_sample/`
     - `data/processed/features_sample/`

4. Runs `pipelines.complex_sample` to generate `reports/analysis_sample/`.
5. Prepares weekly streaming drops with `streaming.sample_prepare_batches`.
6. Starts `streaming.sample_stream` to write per drop reports under `reports/streaming_sample/YYYYMMDD/`.

---

## 5. Running Individual Stages

### 5.1 Ingestion only

Full dataset:

```bash
python -m pipelines.ingest_full
```

Sample dataset:

```bash
python -m pipelines.ingest_sample
```

### 5.2 Train TF-IDF model only

Full model:

```bash
python -m pipelines.train_full
```

Sample model:

```bash
python -m pipelines.train_sample
```

### 5.3 Build full CSR index only

```bash
python -m pipelines.build_full_index
```

Optional flags:

- `--features-path` to point to a non default features directory
- `--out-dir` to change the index output directory
- `--max-docs` to cap the number of rows for experiments

### 5.4 Run complex analytics only

Full:

```bash
python -m pipelines.complex_full
```

Sample:

```bash
python -m pipelines.complex_sample
```

### 5.5 Run streaming pipelines

Sample streaming:

```bash
python -m streaming.sample_prepare_batches   --start-date "$(date +%Y-%m-%d)"   --interval-seconds 1   --no-sleep   --overwrite

python -m streaming.sample_stream
```

Full streaming:

```bash
python -m streaming.full_stream --once
```

### 5.6 Start the web application

With at least the sample pipeline in place:

```bash
export FLASK_APP=app:create_app
export FLASK_ENV=development

flask run --host 0.0.0.0 --port 5000
```

Or:

```bash
python -m app.server
```

Then visit:

- `http://localhost:5000/` for search
- `http://localhost:5000/standard` for standard queries
- `http://localhost:5000/complex` for complex analytics
- `http://localhost:5000/streaming` for streaming reports

---

## 6. Validating the Output

### 6.1 Validate ingestion

```bash
python - << 'EOF'
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("data/processed/arxiv_full")
df.printSchema()
EOF
```

Key fields should include:

- `id`, `id_base`, `paper_id`
- `title`, `abstract`
- `categories`, `primary_category`, `categories_list`
- `year`, `update_date`, `submitted_date`
- `authors_list`, `num_authors`
- `abstract_len`, `title_len`
- `has_doi`, `n_versions`

### 6.2 Validate models

```bash
ls data/models/tfidf_full
ls data/models/tfidf_sample
```

### 6.3 Validate feature Parquet

```bash
python - << 'EOF'
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("data/processed/features_full")
df.select("id_base", "paper_id", "year", "features").show(5, truncate=False)
EOF
```

### 6.4 Validate full index

```bash
ls data/processed/full_index
```

Expected files include the CSR matrix and metadata arrays.

---

## 7. Troubleshooting

- Out of memory during ingestion or training  
  - Reduce `spark.sql.shuffle.partitions` in `engine.utils.spark_utils`.
  - Increase driver memory with `SPARK_DRIVER_MEMORY`.
  - Use the sample pipeline instead of the full pipeline.

- Index build is too slow or heavy  
  - Use `--max-docs` for smaller experiments.
  - Run on a machine with more memory.

- Streaming job appears idle  
  - Check that new files exist in `data/stream/incoming_sample/` or `data/stream/incoming/`.
  - Inspect Spark logs for checkpoint and trigger status.

- Web app shows empty dropdowns  
  - Confirm that the corresponding `reports/` directories contain CSV files.
  - Make sure batch or streaming jobs were run before starting the app.

---

## 8. Summary

To fully reproduce the Sparxiv experiments and figures:

1. Install dependencies and ensure Python, Java, and disk space are adequate.
2. Download the Kaggle arXiv metadata snapshot, or create the sample JSONL.
3. Run `bash run_sample.sh` for a fast end to end test.
4. Run `bash run.sh` and `python -m pipelines.build_full_index` for the full dataset.
5. Launch the Flask app to explore search results and analytics reports.

After these steps you will have:

- Ingested Parquet for full and sample datasets.
- TF-IDF models and feature tables.
- A full CSR search index.
- Batch and streaming analytics in CSV and PNG form.
- An interactive web interface over search and reports.
