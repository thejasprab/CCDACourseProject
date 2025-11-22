# Sparxiv: A Spark-Based Recommender System for arXiv

### Team Members
1. **Ali Khaleghi Rahimian** — akhalegh@charlotte.edu  
2. **Kiyoung Kim** — kkim43@charlotte.edu  
3. **Thejas Prabakaran** — tprabaka@charlotte.edu  

---

## Overview

Sparxiv is an end-to-end **large-scale paper recommender system** built on top of **Apache Spark** and the **Cornell arXiv metadata** dataset (via Kaggle).  
It integrates **batch processing**, **streaming analytics**, and a **content-based recommendation engine** into a unified, scalable workflow.

The system supports the full data lifecycle:

- **Batch ingestion pipeline**  
  Converts raw JSON/JSONL metadata into **cleaned, normalized, partitioned Parquet** using Spark-based transformations.

- **Feature engineering & transformations**  
  Performs text normalization, metadata cleanup, abstract filtering, and field standardization to prepare high-quality input for downstream analytics.

- **Advanced Spark SQL analytics**  
  Computes rich insights such as:
  - category co-occurrence networks  
  - temporal topic trends  
  - author collaboration patterns  
  - DOI/version coverage  
  - abstract length and readability patterns  
  - category migration and productivity trends  

- **Streaming analytics (sample and full)**  
  Uses **Structured Streaming** to process simulated **weekly metadata drops**, generating per-batch reports on categories, submission trends, and DOI statistics.

- **Content-based recommendation engine**  
  Implements a scalable **TF-IDF + cosine similarity** model for paper-to-paper and free-text similarity search:
  - **Spark ML** for training and offline feature generation
  - **In-memory / CSR-based indices** for fast query-time inference (sample + full)

- **Interactive web application**  
  A lightweight Flask UI that provides:
  - a **real-time similarity search** interface  
  - a browser for **complex analytics reports** (CSV/plots)

Two execution modes are supported:

- **Sample workflow** (~50k records): fast demos, debugging, CI runs  
- **Full workflow** (1.7M–2.8M+ records): realistic, high-volume experiments with offline-built search indices

---

## Dataset

- **Source**: Kaggle → `Cornell-University/arxiv`  
- **Format**: JSON Lines (**JSONL**, one record per line)  
- **Size**: ~4–6 GB (metadata, depending on snapshot)  
- **Expected raw paths**:
  - Full snapshot: `data/raw/arxiv-metadata-oai-snapshot.json`
  - Sample JSONL: `data/sample/arxiv-sample.jsonl` (generated from the full file)

The helper script `streaming/kaggle_downloader.py` uses **KaggleHub** to download the dataset into the project structure and optionally writes a **head-N JSONL sample**.

---

## Repository Structure

High-level layout:

```text
ccda-course-project_v2/
├─ run.sh                          # Full pipeline: ingest + train + complex analytics
├─ run_sample.sh                   # SAMPLE pipeline: ingest + train + complex analytics + streaming (sample)
├─ requirements.txt
├─ engine/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ spark_utils.py            # SparkSession factory tuned for low-memory envs
│  │  ├─ io_utils.py               # JSON writer, directory helpers
│  │  └─ misc.py                   # Simple project logger
│  ├─ ml/
│  │  ├─ featurization.py          # RegexTokenizer + stopwords + TF-IDF + L2 Normalizer
│  │  ├─ train.py                  # Train TF-IDF model + write features parquet
│  │  ├─ model_loader.py           # Load trained model + features by mode (sample/full)
│  │  └─ __init__.py
│  ├─ search/
│  │  ├─ similarity.py             # Cosine via sparse dot product + exact Top-K (Spark)
│  │  ├─ vectorize.py              # Vectorize free-text query via Spark TF-IDF pipeline
│  │  └─ search_engine.py          # High-level SearchEngine wrapper (sample: in-memory; full: CSR index)
│  ├─ complex/
│  │  └─ complex_queries.py        # 10 complex Spark SQL / DataFrame analyses
│  └─ data/
│     ├─ ingestion.py              # Ingestion JSON/JSONL → cleaned Parquet (used by pipelines)
│     └─ transformations.py        # Shared transforms used in batch + streaming
├─ pipelines/
│  ├─ ingest_sample.py             # Ingest sample JSONL → Parquet
│  ├─ ingest_full.py               # Ingest full snapshot → Parquet
│  ├─ train_sample.py              # Train TF-IDF on sample parquet
│  ├─ train_full.py                # Train TF-IDF on full parquet
│  ├─ complex_sample.py            # Run 10 complex analyses on sample
│  ├─ complex_full.py              # Run 10 complex analyses on full
│  └─ build_full_index.py          # Build CSR index + metadata arrays for FULL TF-IDF features
├─ streaming/
│  ├─ kaggle_downloader.py         # Download Kaggle dataset + optional sample JSONL
│  ├─ sample_prepare_batches.py    # Generate weekly-dated sample drops for streaming
│  ├─ sample_stream.py             # Structured Streaming (sample weekly drops) → per-drop reports
│  ├─ full_stream.py               # Structured Streaming (full weekly snapshots) → per-drop reports
│  └─ merge_diff.py                # Write only *new* papers between two parquet snapshots
├─ app/
│  ├─ __init__.py                  # Flask app factory (create_app)
│  ├─ config.py                    # Small Settings dataclass (default_mode=sample)
│  ├─ server.py                    # Flask routes/views (search UI + complex analytics browser)
│  ├─ services/
│  │  ├─ spark_session.py          # Shared SparkSession for the web app
│  │  ├─ search_service.py         # Thin wrapper over SearchEngine (sample/full)
│  │  ├─ filters_service.py        # Helper to list popular categories
│  │  └─ complex_service.py        # Helpers to list/load complex analytics CSV reports
│  ├─ templates/
│  │  ├─ base.html                 # Layout + nav
│  │  ├─ index.html                # Similarity search UI
│  │  └─ complex.html              # Complex analytics browser
│  └─ static/
│     ├─ style.css                 # Minimal dark theme styling
│     └─ app.js                    # Small UX helpers
├─ data/
│  ├─ raw/                         # Raw Kaggle snapshot(s) (ignored by Git)
│  ├─ sample/                      # Sample JSONL (ignored by Git)
│  ├─ processed/                   # Batch parquet outputs (ingest + features + full_index) (ignored)
│  └─ stream/
│     ├─ incoming_sample/          # Sample weekly JSONL drops
│     ├─ incoming/                 # Full weekly JSON/JSONL drops
│     └─ checkpoints_*             # Structured Streaming checkpoints
├─ reports/
│  ├─ analysis_sample/             # Batch complex analytics (sample)
│  ├─ analysis_full/               # Batch complex analytics (full)
│  ├─ streaming_sample/YYYYMMDD/   # Per-drop streaming reports (sample)
│  └─ streaming_full/YYYYMMDD/     # Per-drop streaming reports (full)
└─ spark-warehouse/                # Local Spark SQL warehouse (ignored)
```

---

## Environment & Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
pyspark==3.5.1
pandas
pyarrow
matplotlib
jupyter
kagglehub
Flask
scipy
```

### Java / Spark

- **Java**: Use **Java 17+** (e.g., Temurin 17).  
- **Memory**: You can override Spark driver/executor memory via environment variables:

```bash
export SPARK_DRIVER_MEMORY=10g
export SPARK_EXECUTOR_MEMORY=10g
```

The helper `engine.utils.spark_utils.get_spark()` also sets:

- Reasonable `spark.sql.shuffle.partitions`, AQE, skew handling, and local spill dir (`data/tmp/spark-local`).
- ZSTD compression for Parquet by default.
- Conservative Parquet vectorized reader behavior to avoid heap explosions.

---

## Quick Start – Sample Pipeline  
(Ingestion → TF-IDF → Complex Analytics → Streaming → Fast Search)

The **sample** pipeline is the easiest way to see everything end-to-end.

From the project root:

```bash
bash run_sample.sh
```

This script:

1. **Ensures sample dataset exists**

   - If `data/sample/arxiv-sample.jsonl` is missing:
     - Downloads Kaggle dataset via:

       ```bash
       python -m streaming.kaggle_downloader --mode sample --sample-size 50000
       ```

     - Writes a head-N JSONL sample.

2. **Ingests sample → Parquet**

   ```bash
   python -m pipelines.ingest_sample
   ```

   - Reads `data/sample/arxiv-sample.jsonl`
   - Runs ingestion/transformations
   - Writes partitioned Parquet to `data/processed/arxiv_sample/`
   - Prints top categories and counts by year

3. **Trains TF-IDF model on sample**

   ```bash
   python -m pipelines.train_sample
   ```

   Under the hood this calls:

   ```python
   train_model(
       input_parquet="data/processed/arxiv_sample",
       model_dir="data/models/tfidf_sample",
       features_out="data/processed/features_sample",
       vocab_size=80000,
       min_df=3,
       use_bigrams=False,
       extra_stopwords_topdf=200,
   )
   ```

   Artifacts:

   - Model: `data/models/tfidf_sample/` (plus `model.json` metadata)
   - Features parquet: `data/processed/features_sample/` with columns such as:
     - `id_base`, `paper_id`, `title`, `abstract`, `categories`, `year`, `features`

4. **Runs complex Spark SQL analytics (sample)**

   ```bash
   python -m pipelines.complex_sample
   ```

   Internally:

   - Reads parquet from `data/processed/arxiv_sample`
   - Registers `papers` + `papers_enriched` views
   - Runs 10 analyses defined in `engine.complex.complex_queries`, including:
     - Category co-occurrence
     - Author collaboration over time
     - Rising/declining topics
     - Abstract readability/length trends
     - DOI vs versions correlation
     - Author productivity & category migration
     - Abstract length vs popularity
     - Weekday submission patterns (if `submitted_date` available)
     - Category stability via versions
   - Writes per-analysis CSV + PNG under:

     ```text
     reports/analysis_sample/
     ```

5. **Prepares weekly sample streaming drops**

   ```bash
   python -m streaming.sample_prepare_batches        --start-date "$(date +%Y-%m-%d)"        --interval-seconds 1        --no-sleep        --overwrite
   ```

   This:

   - Reads `data/sample/arxiv-sample.jsonl`
   - Writes 5 weekly-dated JSONL files (head 10k, 20k, 30k, 40k, 50k lines) into:

     ```text
     data/stream/incoming_sample/arxiv-sample-YYYYMMDD.jsonl
     ```

6. **Starts sample Structured Streaming job**

   ```bash
   python -m streaming.sample_stream
   ```

   - Watches `data/stream/incoming_sample/` for `arxiv-sample-*.jsonl`
   - For each new file (micro-batch):

     - Applies `engine.data.transformations.transform_all`
     - Emits per-drop CSVs + PNGs:
       - `reports/streaming_sample/YYYYMMDD/by_year.csv`
       - `reports/streaming_sample/YYYYMMDD/top_categories.csv`
       - `reports/streaming_sample/YYYYMMDD/doi_rate_by_year.csv`
       - `.../papers_per_year.png`, `top_categories.png`, `doi_rate_by_year.png`

   - Runs until you **Ctrl+C**.

7. **Fast sample search index (built implicitly on first web-app use)**

   When the web app starts in **sample** mode, the `SearchEngine`:

   - Loads `data/models/tfidf_sample/` and `data/processed/features_sample/` with Spark.
   - Pulls the sample features into memory as Python `SparseVector` objects + metadata.
   - Serves queries via:
     - Spark TF-IDF transform for **1 query vector**
     - pure Python cosine similarity over the in-memory sample corpus.

   Result: **very fast inference for sample dataset**, no large Spark jobs per query.

---

## Quick Start – Full Pipeline  
(Ingestion → TF-IDF → Complex Analytics → Full TF-IDF Index)

For the full dataset:

```bash
bash run.sh
```

This script:

1. **Ensures full raw dataset exists**

   - If `data/raw/arxiv-metadata-oai-snapshot.json` is missing:
     - Runs:

       ```bash
       python -m streaming.kaggle_downloader --mode full
       ```

     - Downloads Kaggle dataset and copies/renames to the expected path.

2. **Ingests full dataset → Parquet**

   ```bash
   python -m pipelines.ingest_full
   ```

   - Reads `data/raw/arxiv-metadata-oai-snapshot.json` (JSONL)
   - Runs ingestion + transformation logic
   - Writes partitioned Parquet to `data/processed/arxiv_full/`

3. **Trains TF-IDF model on full dataset**

   ```bash
   python -m pipelines.train_full
   ```

   Internally:

   ```python
   train_model(
       input_parquet="data/processed/arxiv_full",
       model_dir="data/models/tfidf_full",
       features_out="data/processed/features_full",
       vocab_size=250000,
       min_df=5,
       use_bigrams=False,
       extra_stopwords_topdf=500,
   )
   ```

   Artifacts:

   - Model: `data/models/tfidf_full/`
   - Features parquet: `data/processed/features_full/`

4. **Runs complex Spark SQL analytics (full)**

   ```bash
   python -m pipelines.complex_full
   ```

   - Same 10 complex queries as sample
   - Outputs to `reports/analysis_full/`

5. *(Optional)* **Full streaming job**

   In `run.sh`, the full streaming job is left **commented out**:

   ```bash
   # python -m streaming.full_stream --once
   ```

   You can enable this if you want to process the latest full “drop” in a one-shot manner, writing reports under:

   ```text
   reports/streaming_full/YYYYMMDD/
   ```

6. **Build full TF-IDF search index (fast full-mode inference)**

   To avoid running a massive Spark cross-join per search query, we build a **SciPy CSR index** and metadata arrays offline, entirely in Python via `pyarrow`:

   ```bash
   python -m pipelines.build_full_index
   ```

   This:

   - Scans `data/processed/features_full/` with `pyarrow.dataset` in batches.
   - Reconstructs TF-IDF sparse vectors from the Spark VectorUDT representation.
   - Builds a global CSR matrix of shape `(num_docs, vocab_dim)`.
   - Saves under `data/processed/full_index/`:

     ```text
     full_index_csr.npz          # CSR sparse TF-IDF matrix
     full_index_ids.npy          # id_base per row
     full_index_paper_ids.npy    # paper_id per row
     full_index_titles.npy       # title per row
     full_index_abstracts.npy    # abstract per row
     full_index_categories.npy   # categories per row
     full_index_years.npy        # year per row
     ```

   At query time in **full mode**, the web app then:

   - Uses Spark TF-IDF pipeline to vectorize the query text to a single sparse vector.
   - Converts it to a dense NumPy float32 vector.
   - Computes `scores = CSR_matrix @ query_vector` in Python (fast sparse matvec).
   - Picks top-K and reads metadata from the preloaded `.npy` arrays.

   Result: **fast full-dataset inference** with no giant Spark cross-join per request.

---

## Web App – TF-IDF Search UI + Analytics Browser

We provide a lightweight Flask app under `app/` that:

1. Exposes a **similarity search UI** over the sample/full TF-IDF models.
2. Lets you **browse complex analytics reports** as interactive tables.

### 1. Ensure prerequisites

Run at least:

- For **sample UI**:
  - `bash run_sample.sh`  

- For **full UI**:
  - `bash run.sh`
  - `python -m pipelines.build_full_index`  

  (full search requires both `features_full` and the `full_index` artifacts).

### 2. Start the Flask app

From the project root:

```bash
export FLASK_APP=app:create_app
export FLASK_ENV=development  # optional for debug / auto-reload

flask run --host 0.0.0.0 --port 5000
```

Alternatively:

```bash
python -m app.server
# or
python app/server.py
```

### 3. Use the UI

Open in a browser:

- **Search UI**: <http://localhost:5000/>

  - Choose dataset: **sample** or **full**
  - Enter **title** and/or **abstract**
  - Choose **Top-K** (1–50)
  - Submit to get Top-K similar papers with:
    - arXiv ID & link (`https://arxiv.org/abs/<paper_id>`)
    - Title, year, categories
    - Cosine similarity score

  Under the hood:

  - **Sample mode:**
    - Spark TF-IDF pipeline transforms query → 1 sparse vector.
    - Python computes cosine similarity against in-memory sample vectors.
  - **Full mode:**
    - Spark TF-IDF pipeline transforms query → 1 sparse vector.
    - Python does CSR matrix–vector multiply using the offline-built full index.
  - Spark is *not* doing a crossJoin per query anymore.

- **Complex analytics browser**: <http://localhost:5000/complex>

  - Choose dataset (sample/full).
  - Select one of the CSV outputs in `reports/analysis_{sample,full}/`.
  - Preview the report in a scrollable HTML table.

### 4. Web app internals

- **Spark session** for the app:
  - Managed by `app.services.spark_session.get_spark_session()`
  - Reuses the same tuned config as batch/streaming.

- **Search**:
  - `app.services.search_service.search_papers`:
    - Caches a `SearchEngine` per mode (`sample`, `full`).
    - Internally uses:
      - `engine.search.search_engine.SearchEngine`  
      - `engine.search.vectorize.vectorize_query` for query → vector  
      - local sample index or full CSR index for similarity.

- **Filters / hints**:
  - `app.services.filters_service.list_primary_categories`:
    - Reads features parquet.
    - Aggregates by category to surface popular labels.

- **Complex reports**:
  - `app.services.complex_service.list_complex_reports(mode)`
  - `app.services.complex_service.load_complex_report(path)`

---

## ML Configuration Knobs

The core training function `engine.ml.train.train_model` supports:

- `vocab_size`  
- `min_df` (min document frequency per term)  
- `use_bigrams` (whether to append bigrams)  
- `extra_stopwords_topdf` (number of top-DF tokens to treat as extra stopwords)  
- `seed` (for deterministic behaviors in extra stopword computation)  

Defaults differ slightly for **sample** vs **full** (see pipelines):

- **Sample**:
  - `vocab_size=80_000`, `min_df=3`, `extra_stopwords_topdf=200`
- **Full**:
  - `vocab_size=250_000`, `min_df=5`, `extra_stopwords_topdf=500`

The pipeline creates:

- `features_norm` as L2-normalized TF-IDF vectors.
- A `features` column in the final `features_*` parquet for use by the search engine and full index builder.

---

## Notes / Troubleshooting

**Hostname / native-hadoop WARNs**

You might see messages like:

```text
WARN Utils: Your hostname resolves to a loopback address...
WARN NativeCodeLoader: Unable to load native-hadoop library...
```

These are typically harmless when running in local mode. To quiet the hostname warning:

```bash
export SPARK_LOCAL_IP=127.0.0.1   # or a specific interface IP
```

**Streaming “No Partition Defined for Window operation!” WARNs**

These are expected when some analyses use a global window (e.g., `NTILE` without partition).  
They can be ignored for small-/medium-sized runs; for very large clusters, you’d want to tune partitioning.

**OOM / slow shuffles on full**

- Increase Spark memory (`SPARK_DRIVER_MEMORY`, `SPARK_EXECUTOR_MEMORY`).  
- Ensure `data/tmp/spark-local/` has enough disk space for spills.  
- Reduce `spark.sql.shuffle.partitions` if you have fewer cores.  
- For streaming, adjust `--max-files-per-trigger` to limit per-batch load.  
- For full search:
  - Make sure `pipelines.build_full_index` completed successfully.
  - If CSR index is too large for RAM, consider:
    - limiting to a subset of years/categories, or  
    - building multiple shard indices.

---

## License

- Code: MIT (or course default license).  
- Dataset metadata: **CC0 (Public Domain)** as per arXiv metadata license.  
  - Individual PDFs/papers may carry different licenses and should be respected.

---

## Acknowledgements

- **arXiv** (Cornell University) for maintaining the dataset and service.  
- **Kaggle & KaggleHub** for hosting and convenient dataset access.  
- **Apache Spark** community for the core engine used throughout this project.
