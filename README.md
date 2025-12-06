# Sparxiv: A Spark based recommender system for arXiv

## Team

1. **Ali Khaleghi Rahimian** - akhalegh@charlotte.edu  
2. **Kiyoung Kim** - kkim43@charlotte.edu  
3. **Thejas Prabakaran** - tprabaka@charlotte.edu  

---

## Overview

Sparxiv is an end to end large scale paper recommender system built on top of **Apache Spark** and the **Cornell arXiv metadata** dataset (via Kaggle).

It combines:

- **Batch ingestion and cleaning** of raw JSON / JSONL metadata into partitioned Parquet  
- **Feature engineering and TF IDF based document vectors**  
- **Standard and complex Spark SQL analytics** on the arXiv corpus  
- **Structured Streaming** over synthetic weekly drops  
- A **content based recommendation engine** for paper similarity search  
- A **Flask web app** for interactive search and report browsing

Two execution modes are supported:

- **Sample mode**: small subset (~50k rows) for fast demos and CI  
- **Full mode**: full snapshot (3M+ records) for realistic experiments

High level and detailed architecture diagrams live in:

- `docs/Sparxiv_SysArchitecture_Simple.webp`  
- `docs/Sparxiv_SysArchitecture.webp`  

---

## Dataset

- **Source**: Kaggle dataset `Cornell-University/arxiv`  
- **Format**: JSON Lines (JSONL), one record per paper version  
- **Typical size**: about 4 to 6 GB of metadata, depending on snapshot

Expected raw input paths:

- Full snapshot JSONL (single big file):  
  - `data/raw/arxiv-metadata-oai-snapshot.json`
- Sample JSONL (derived head N subset):  
  - `data/sample/arxiv-sample.jsonl`

The helper:

```bash
python -m streaming.kaggle_downloader --mode sample --sample-size 50000
python -m streaming.kaggle_downloader --mode full
```

downloads the Kaggle dataset and writes:

- `data/raw/arxiv-metadata-oai-snapshot.json`  
- optionally a `data/sample/arxiv-sample.jsonl` head sample

---

## Environment and requirements

### Python

Install dependencies:

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

Python 3 with a reasonably recent minor version is recommended.

### Java / Spark

PySpark 3.5.1 expects a modern Java runtime.

- Install **Java 17+** (for example Temurin 17)
- Optionally tune Spark memory via environment variables:

```bash
export SPARK_DRIVER_MEMORY=10g
export SPARK_EXECUTOR_MEMORY=10g
```

The utility `engine.utils.spark_utils.get_spark()` configures:

- UTC session timezone  
- Reasonable defaults for shuffles and partitions  
- Conservative Parquet vectorized reading to avoid big heap allocations  
- Local spill directory at `data/tmp/spark-local`  
- ZSTD compression for Parquet  

---

## Project layout

Root level:

```text
sparxiv/
├─ run.sh                     # Full pipeline entry point
├─ run_sample.sh              # Sample pipeline entry point
├─ requirements.txt
├─ README.md                  # This file
├─ engine/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ misc.py              # Logging helper
│  │  ├─ spark_utils.py       # SparkSession factory and tuning
│  │  ├─ io_utils.py          # Simple IO helpers (dirs, JSON)
│  │  └─ __init__.py
│  ├─ ml/
│  │  ├─ featurization.py     # Tokenization, stopwords, TF IDF, normalization
│  │  ├─ train.py             # Train TF IDF pipeline and write features Parquet
│  │  ├─ model_loader.py      # Load trained pipeline + features per mode
│  │  └─ __init__.py
│  ├─ search/
│  │  ├─ similarity.py        # Sparse cosine via UDF + exact Top K in Spark
│  │  ├─ vectorize.py         # Query text to TF IDF vector via Spark ML
│  │  └─ search_engine.py     # High level SearchEngine (sample vs full)
│  ├─ complex/
│  │  └─ complex_queries.py   # Complex Spark SQL / DataFrame analyses
│  └─ data/
│     ├─ ingestion.py         # Raw JSON(L) to cleaned Parquet
│     └─ transformations.py   # Shared field cleanup and normalization
├─ pipelines/
│  ├─ ingest_sample.py        # Ingest sample JSONL to Parquet
│  ├─ ingest_full.py          # Ingest full snapshot to Parquet
│  ├─ train_sample.py         # Train TF IDF on sample Parquet
│  ├─ train_full.py           # Train TF IDF on full Parquet
│  ├─ complex_sample.py       # Run complex analyses on sample Parquet
│  ├─ complex_full.py         # Run complex analyses on full Parquet
│  └─ build_full_index.py     # Build CSR index + metadata arrays for full mode
├─ streaming/
│  ├─ kaggle_downloader.py    # Download dataset and optionally create sample JSONL
│  ├─ sample_prepare_batches.py  # Split sample into synthetic weekly JSONL drops
│  ├─ sample_stream.py        # Structured Streaming over sample drops
│  ├─ full_stream.py          # Structured Streaming over full weekly snapshots
│  └─ merge_diff.py           # Parquet level incremental merge (new papers only)
├─ app/
│  ├─ __init__.py             # Flask app factory
│  ├─ config.py               # Settings dataclass (default_mode etc)
│  ├─ server.py               # Routes and views
│  ├─ services/
│  │  ├─ spark_session.py     # Shared SparkSession for the web app
│  │  ├─ search_service.py    # SearchEngine wrapper + caching
│  │  ├─ filters_service.py   # Category listing helpers
│  │  ├─ standard_service.py  # Standard query report loader
│  │  ├─ complex_service.py   # Complex analytics report loader
│  │  └─ streaming_service.py # Streaming report loader
│  ├─ templates/
│  │  ├─ base.html            # Layout and navigation
│  │  ├─ index.html           # Similarity search UI
│  │  ├─ standard.html        # Standard query reports browser
│  │  ├─ complex.html         # Complex analytics browser
│  │  └─ streaming.html       # Streaming reports browser
│  └─ static/
│     ├─ style.css            # Minimal dark theme
│     ├─ app.js               # Small UX helpers
│     └─ favicon.svg
├─ docs/
│  ├─ dataset_overview.md     # Explanation of raw fields and transforms
│  ├─ methodology.md          # Design choices across pipelines and search
│  ├─ reproduction_guide.md   # Step by step reproduction instructions
│  ├─ limitations.md          # Known limitations and caveats
│  ├─ results.md              # Summary of key metrics and visual findings
│  ├─ Sparxiv_SysArchitecture_Simple.webp
│  └─ Sparxiv_SysArchitecture.webp
├─ reports/
│  ├─ standard_queries_sample/  # Standard query CSVs for sample mode
│  ├─ standard_queries_full/    # Standard query CSVs for full mode
│  ├─ analysis_sample/          # Complex analysis CSVs and figures (sample)
│  ├─ analysis_full/            # Complex analysis CSVs and figures (full)
│  ├─ streaming_sample/YYYYMMDD/  # Per batch streaming reports (sample)
│  └─ streaming_full/YYYYMMDD/    # Per batch streaming reports (full)
└─ spark-warehouse/            # Local Spark SQL warehouse (created at runtime)
```

Note: the `data/` directory (raw JSON, sample JSONL, processed parquet, stream inputs, checkpoints) is created at runtime and is typically ignored by version control.

---

## Core components

### 1. Ingestion and transformations

`engine.data.ingestion.run_ingestion` reads JSON / JSONL and writes cleaned, partitioned Parquet:

- Parses ids, titles, abstracts, categories  
- Extracts primary category and normalized category list  
- Normalizes authors and `authors_parsed`  
- Parses dates (submitted, updated) to proper timestamp and year columns  
- Computes helper columns used downstream (version count, DOI flags etc)

Entry points:

- `pipelines/ingest_sample.py`  
- `pipelines/ingest_full.py`  

Defaults:

- Sample output: `data/processed/arxiv_sample` (partitioned by year)  
- Full output: `data/processed/arxiv_full`  

### 2. Feature engineering and TF IDF model

`engine.ml.featurization` and `engine.ml.train` build a Spark ML pipeline that:

- Concatenates title and abstract to a single text field  
- Tokenizes text with `RegexTokenizer`  
- Removes standard and custom stopwords  
- Applies `HashingTF` + `IDF` (with configurable vocabulary size and `min_df`)  
- L2 normalizes the resulting vectors into a `features_norm` column

Training scripts:

- `pipelines/train_sample.py`
- `pipelines/train_full.py`

They call `train_model(...)` with different settings:

- **Sample**: smaller vocabulary, lower `min_df`  
- **Full**: larger vocabulary, higher `min_df`, more extra stopwords

Artifacts:

- Trained pipeline: `data/models/tfidf_sample/` or `data/models/tfidf_full/`  
- Features parquet: `data/processed/features_sample/` or `data/processed/features_full/`  
  - Includes `id_base`, `paper_id`, `title`, `abstract`, `categories`, `year`, `features`

`engine.ml.model_loader.load_model_and_features` is used by both CLI scripts and the web app to load these artifacts.

### 3. Search engine

The core search logic lives in:

- `engine.search.similarity`  
- `engine.search.vectorize`  
- `engine.search.search_engine.SearchEngine`

Workflow:

1. Query text (title + abstract) is vectorized via the trained Spark ML pipeline  
2. For **sample mode**:
   - All document vectors are pulled into Python once
   - Cosine similarity is computed in pure Python / NumPy against in memory vectors  
3. For **full mode**:
   - Offline script `pipelines/build_full_index.py` constructs a SciPy CSR matrix:
     - `full_index_csr.npz`               (document term matrix)  
     - `full_index_ids.npy`               (internal id)  
     - `full_index_paper_ids.npy`         (arXiv ids)  
     - `full_index_titles.npy`  
     - `full_index_abstracts.npy`  
     - `full_index_categories.npy`  
     - `full_index_years.npy`  
   - At query time the SearchEngine:
     - Vectorizes the query with Spark  
     - Converts it to a dense NumPy vector  
     - Computes `scores = CSR_matrix @ query_vector`  
     - Returns Top K neighbors with metadata

There is also a fallback Spark based search path that does brute force cosine in Spark if the local index is missing.

### 4. Standard analytics

"Standard queries" compute basic but useful statistics on the ingested data, for example:

- Submissions per year (`by_year.csv`)  
- Top categories (`top_categories.csv`)  
- Category year matrix (`category_year_matrix.csv`)  
- DOI coverage by year (`doi_rate_by_year.csv`)  
- Text length summary (`text_length_summary.csv`)  
- Top authors (`top_authors.csv`)  
- Version count distributions (`version_count_hist.csv`)  
- Category Pareto breakdowns and completeness statistics  

Outputs live under:

- `reports/standard_queries_sample/`  
- `reports/standard_queries_full/`

These CSVs and any matching PNG figures are browsed via the `/standard` route in the web app.

### 5. Complex analytics

`engine.complex.complex_queries` contains more advanced analyses that make heavier use of Spark SQL and plotting:

Examples include:

- Category co occurrence networks  
- Author collaboration over time  
- Rising and declining topic categories  
- DOI vs version count correlation  
- Abstract length vs number of versions  
- Category migration and author lifecycle patterns  
- Lexical richness by year  
- Additional auxiliary views over enriched metadata

The CLI frontends:

- `pipelines/complex_sample.py`  
- `pipelines/complex_full.py`  

write results to:

- `reports/analysis_sample/`  
- `reports/analysis_full/`  

Each analysis usually produces:

- A CSV with aggregated metrics  
- Optional PNG figures (bar plots, line plots, scatter plots, single value panels)

These are browsed with the `/complex` route of the web app.

### 6. Streaming analytics

The streaming subsystem simulates weekly metadata drops.

Key pieces:

- `streaming/sample_prepare_batches.py`
  - Reads `data/sample/arxiv-sample.jsonl`
  - Writes synthetic weekly JSONL drops into `data/stream/incoming_sample/`
- `streaming/sample_stream.py`
  - Structured Streaming over `data/stream/incoming_sample/`
  - For each new file:
    - Applies shared transformations
    - Computes per date reports:
      - `by_year.csv`
      - `top_categories.csv`
      - `doi_rate_by_year.csv`
    - Writes CSVs and PNGs under `reports/streaming_sample/YYYYMMDD/`
- `streaming/full_stream.py`
  - Same idea as `sample_stream.py`, but for full snapshots under `data/stream/incoming/`
  - Writes to `reports/streaming_full/YYYYMMDD/`
- `streaming/merge_diff.py`
  - Given two parquet snapshots, writes only the newly added papers

The `/streaming` route in the web app lets you:

- List available incoming snapshots  
- Pick a date stamp (YYYYMMDD)  
- Browse the generated streaming CSVs and figures for that snapshot  

### 7. Web application

The Flask app lives under `app/` and is created via the factory in `app/__init__.py`.

Services:

- `app.services.spark_session.get_spark_session()`
  - Singleton SparkSession for the web app
- `app.services.search_service`
  - Caches one `SearchEngine` per mode (`sample` or `full`)
  - Exposes `search_papers(...)` for views
- `app.services.filters_service`
  - Computes popular categories from features parquet
- `app.services.standard_service`
  - Lists and loads standard query CSVs and figures
- `app.services.complex_service`
  - Lists and loads complex analytics CSVs and figures
- `app.services.streaming_service`
  - Lists and loads streaming snapshots and their reports

Routes (in `app/server.py`):

- `/`  
  Similarity search UI. Lets you:
  - Choose dataset (sample or full)
  - Enter title and/or abstract
  - Choose Top K (1 to 50)
  - Get ranked results with arXiv link, title, year, categories, score

- `/standard`  
  Standard analytics dashboard. Lets you:
  - Choose dataset (sample or full)
  - Pick a standard query CSV
  - Preview table content
  - Optionally choose a related figure if available

- `/complex`  
  Complex analytics dashboard. Similar to `/standard`, but over complex analysis outputs.

- `/streaming`  
  Streaming analytics browser:
  - Select dataset mode (sample or full)
  - Select a date stamp for which streaming reports exist
  - Inspect CSV tables and figures generated by the streaming jobs

---

## Quick start: sample pipeline

The sample pipeline gives a full end to end run on a manageable subset.

From the project root:

```bash
bash run_sample.sh
```

This script coordinates:

1. **Sample dataset check / download**

   - If `data/sample/arxiv-sample.jsonl` is missing:
     - Runs `python -m streaming.kaggle_downloader --mode sample --sample-size 50000`
     - Downloads the full Kaggle snapshot and writes a head N JSONL sample

2. **Ingest sample to Parquet**

   ```bash
   python -m pipelines.ingest_sample
   ```

   - Input: `data/sample/arxiv-sample.jsonl`  
   - Output: `data/processed/arxiv_sample/` (partitioned by year)

3. **Train TF IDF on sample**

   ```bash
   python -m pipelines.train_sample
   ```

   Produces:

   - `data/models/tfidf_sample/`
   - `data/processed/features_sample/`

4. **Run complex analytics on sample**

   ```bash
   python -m pipelines.complex_sample
   ```

   Writes CSVs and figures under `reports/analysis_sample/`.

5. **Prepare streaming batches for sample**

   Example invocation (run automatically in the script with current date):

   ```bash
   python -m streaming.sample_prepare_batches      --start-date "$(date +%Y-%m-%d)"      --interval-seconds 1      --no-sleep      --overwrite
   ```

   Writes:

   - `data/stream/incoming_sample/arxiv-sample-YYYYMMDD*.jsonl` (5 weekly drops)

6. **Run sample streaming job**

   ```bash
   python -m streaming.sample_stream
   ```

   - Watches `data/stream/incoming_sample/`
   - For each new file:
     - Applies transformations
     - Writes CSVs and PNGs into `reports/streaming_sample/YYYYMMDD/`
   - Runs until interrupted with Ctrl+C

7. **Search index for sample**

   The sample search index is built lazily:

   - First time the web app receives a sample mode search request, the SearchEngine:
     - Loads `tfidf_sample` model and `features_sample` parquet
     - Pulls feature vectors into memory as sparse objects and caches them

---

## Quick start: full pipeline

The full pipeline runs on the entire dataset.

From the project root:

```bash
bash run.sh
```

High level steps:

1. **Ensure full raw dataset exists**

   - If `data/raw/arxiv-metadata-oai-snapshot.json` is missing:
     - Runs `python -m streaming.kaggle_downloader --mode full`
     - Downloads the Kaggle dataset and writes the expected raw path

2. **Ingest full JSONL to Parquet**

   ```bash
   python -m pipelines.ingest_full
   ```

   - Input: `data/raw/arxiv-metadata-oai-snapshot.json`
   - Output: `data/processed/arxiv_full/`

3. **Train TF IDF on full parquet**

   ```bash
   python -m pipelines.train_full
   ```

   Produces:

   - `data/models/tfidf_full/`
   - `data/processed/features_full/`

4. **Run complex analytics on full**

   ```bash
   python -m pipelines.complex_full
   ```

   Writes CSVs and figures under `reports/analysis_full/`.

5. **Optional: one shot full streaming**

   `run.sh` contains a commented line:

   ```bash
   # python -m streaming.full_stream --once
   ```

   When enabled, this:

   - Processes all current files in `data/stream/incoming/`  
   - Writes streaming reports under `reports/streaming_full/YYYYMMDD/`  

6. **Build full CSR index for fast search**

   ```bash
   python -m pipelines.build_full_index
   ```

   This:

   - Scans `data/processed/features_full/` in batches with `pyarrow.dataset`
   - Reconstructs sparse vectors
   - Builds global CSR matrix and metadata arrays
   - Saves them under `data/processed/full_index/`

   After this step, full mode search is handled via fast sparse matrix vector multiplication instead of Spark cross joins.

---

## Running the web app

Prerequisites:

- For **sample mode only**:
  - `bash run_sample.sh`
- For **full mode**:
  - `bash run.sh`
  - `python -m pipelines.build_full_index` (for fast full search)

Start the app from the project root:

```bash
export FLASK_APP=app:create_app
export FLASK_ENV=development  # optional

flask run --host 0.0.0.0 --port 5000
```

Alternative:

```bash
python -m app.server
# or
python app/server.py
```

Open:

- Search UI: `http://localhost:5000/`
- Standard queries: `http://localhost:5000/standard`
- Complex analytics: `http://localhost:5000/complex`
- Streaming reports: `http://localhost:5000/streaming`

---

## ML configuration knobs

The main training function `engine.ml.train.train_model` exposes:

- `vocab_size`  
- `min_df`  
- `use_bigrams`  
- `extra_stopwords_topdf` (number of high document frequency terms to treat as extra stopwords)  
- `seed` for reproducible extra stopword selection  

Defaults:

- **Sample**:
  - `vocab_size = 80_000`
  - `min_df = 3`
  - `extra_stopwords_topdf = 200`
- **Full**:
  - `vocab_size = 250_000`
  - `min_df = 5`
  - `extra_stopwords_topdf = 500`

The pipeline writes:

- `features_norm` as normalized vectors  
- A `features` column consumed by the search engine and index builder

---

## Troubleshooting

### Spark warnings

Messages like:

- hostname resolves to loopback  
- unable to load native Hadoop library  

are benign in local mode. To reduce hostname related noise:

```bash
export SPARK_LOCAL_IP=127.0.0.1
```

### Memory issues or slow shuffles on full mode

If you see out of memory errors or very slow shuffles:

- Increase `SPARK_DRIVER_MEMORY` and `SPARK_EXECUTOR_MEMORY`  
- Ensure `data/tmp/spark-local/` has enough disk space  
- Reduce `spark.sql.shuffle.partitions` for machines with few cores  
- For streaming, reduce per batch load with `--max-files-per-trigger`  
- For full mode search:
  - Confirm `pipelines.build_full_index` completed successfully
  - If the CSR index does not fit in RAM, consider:
    - Filtering by years or categories
    - Building multiple smaller shard indices

### Missing reports in the web app

- If `/complex` shows no reports:
  - Run `pipelines.complex_sample.py` or `pipelines.complex_full.py`
- If `/standard` shows no reports:
  - Make sure standard query pipelines have been run (or that sample outputs exist under `reports/standard_queries_*`)
- If `/streaming` shows no stamps:
  - Check that `sample_stream.py` or `full_stream.py` have been run
  - Verify that `reports/streaming_*` directories exist and contain CSVs

---

## Additional documentation

See the `docs/` folder for more detailed write ups:

- [`docs/dataset_overview.md`](docs/dataset_overview.md)  
  Field level documentation of the raw Kaggle dataset and derived columns.

- [`docs/methodology.md`](docs/methodology.md)  
  Rationale behind data cleaning, feature engineering, query design, and evaluation.

- [`docs/reproduction_guide.md`](docs/reproduction_guide.md)  
  Step by step instructions for reproducing the experiments and figures.

- [`docs/limitations.md`](docs/limitations.md)  
  Discussion of dataset biases, modeling limitations, and system constraints.

- [`docs/results.md`](docs/results.md)  
  Consolidated experimental results and high level interpretation.

---

## License and data usage

- Code: MIT or course default license (see repository policy if provided).  
- Dataset: arXiv metadata is released under a CC0 public domain dedication.  
  Individual paper PDFs can have their own licenses and must be respected.

---

## Acknowledgements

- **arXiv** (Cornell University) for providing the metadata and service  
- **Kaggle / KaggleHub** for publicly hosting the dataset and client tools  
- **Apache Spark** community for the execution engine used across batch, streaming, and ML
