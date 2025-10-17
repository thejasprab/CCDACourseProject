# ITCS 6190/8190 – Cloud Computing for Data Analysis
## Course Project: Data Analysis with Apache Spark (arXiv on Kaggle)

### Team Members
1. **Ali Khaleghi Rahimian** — akhalegh@charlotte.edu  
2. **Kiyoung Kim** — kkim43@charlotte.edu  
3. **Thejas Prabakaran** — tprabaka@charlotte.edu

---

## Overview
This project implements a **Spark-based data analysis pipeline** for the **Cornell arXiv metadata** dataset from Kaggle. We focus on:
- **Data ingestion** (JSON/JSONL → cleaned & partitioned Parquet)
- **Transformations** (text cleanup, field standardization, simple quality filters)
- **Exploratory Data Analysis (EDA)** (counts, trends, top categories/authors, DOI coverage, etc.)

We provide both a **sample workflow** (fast, 50k records for PRs/demos) and a **full workflow** (≈1.7M+ records; current snapshot ~2.85M rows after quality filters). All steps are designed to run in **GitHub Codespaces** or any local Spark environment.

---

## Dataset
- **Source**: Kaggle → *Cornell-University/arxiv*  
- **Contents**: Metadata of millions of arXiv papers (id, title, abstract, categories, versions, authors, DOI, etc.)  
- **Format**: JSON Lines (one record per line)  
- **Size**: 4–6 GB (metadata JSON), growing with updates

We use `kagglehub` to download the dataset and (optionally) create a small **JSONL sample** for quick iteration.

> **Note**: We do **not** commit the full raw dataset to the repo. Only a small sample (if needed) is kept under `data/sample/` for testing/demo.

---

## Repository Structure
```
.
├─ scripts/
│  └─ download_arxiv.py        # Download full Kaggle dataset + write sample JSONL
├─ src/
│  ├─ ingestion.py             # Batch ingestion (JSON/JSONL → Parquet)
│  ├─ transformations.py       # Field cleanup & derived columns
│  └─ utils.py                 # Spark session + helper utilities
├─ notebooks/
│  └─ eda_week8.py             # Comprehensive EDA (writes CSVs + PNGs)
├─ data/
│  ├─ raw/                     # Full raw file(s) (ignored by Git)
│  ├─ sample/                  # Tiny JSONL sample for PR/demo
│  └─ processed/               # Parquet outputs (ignored by Git)
├─ reports/
│  ├─ sample/                  # EDA outputs for sample run (CSV/PNG)
│  └─ full/                    # EDA outputs for full run (CSV/PNG)
├─ run.sh                      # One-command FULL pipeline (ingest + EDA)
├─ run_sample.sh               # One-command SAMPLE pipeline (ingest + EDA)
├─ requirements.txt
└─ README.md
```

---

## Environment & Requirements
Install Python deps (Codespaces or local):
```bash
pip install -r requirements.txt
```
`requirements.txt`:
```
pyspark==3.5.1
pandas
pyarrow
matplotlib
jupyter
kagglehub
```

> If needed, you can increase Spark resources by editing `src/utils.py → get_spark()` or by exporting memory variables shown below.

---

## Quick Start

### A) Full Pipeline (ingestion → EDA)
Runs end-to-end on the full dataset. The script will download the dataset if missing, write partitioned Parquet, and generate CSV/PNG artifacts under `reports/full/`.

```bash
bash run.sh
```

**Key settings in `run.sh`**
- Project-local spill dir: `data/tmp/spark-local`
- Driver/executor heap: **8g**
- Small input/target partition sizes (8 MB) + AQE
- Write aligned by `year` (`--repartition 200`)

You can override memory without editing the script:
```bash
export SPARK_DRIVER_MEMORY=10g
export SPARK_EXECUTOR_MEMORY=10g
bash run.sh
```

### B) Sample Pipeline (ingestion → EDA)
Fast path using a ~50k-line JSONL sample. Artifacts are written to `reports/sample/`.

```bash
bash run_sample.sh
```

**What `run_sample.sh` does**
- Ensures `data/sample/arxiv-sample.jsonl` exists (creates via KaggleHub if needed)
- Ingests sample → `data/processed/arxiv_parquet/`
- Generates EDA to `reports/sample/`
- Uses smaller resources (default 4g driver/executor, fewer shuffle partitions)

---

## Manual Usage (Advanced)

### 1) Download Dataset (+ optional Sample)
```bash
# full dataset into data/raw/
python scripts/download_arxiv.py --sample 0

# or also create a 50k-line JSONL sample for quick EDA
python scripts/download_arxiv.py --sample 50000
# → data/raw/arxiv-metadata-oai-snapshot.json
# → data/sample/arxiv-sample.jsonl
```

### 2) Ingestion (JSON/JSONL → Parquet)

**Sample (fast, for PR/demo)**
```bash
python -m src.ingestion   --input data/sample/arxiv-sample.jsonl   --output data/processed/arxiv_parquet   --partition-by year   --repartition 64
```

**Full Dataset**
```bash
# IMPORTANT: Do NOT use --multiline for the Kaggle JSON (it's JSONL).
python -m src.ingestion   --input data/raw/arxiv-metadata-oai-snapshot.json   --output data/processed/arxiv_full   --partition-by year   --repartition 200
```

> If you accidentally used `--multiline` on JSONL earlier, delete your old output with:
> ```bash
> rm -rf data/processed/arxiv_full
> ```
> and re-run ingestion without `--multiline`.

### 3) EDA (CSV Tables + PNG Charts)

**Sample EDA (writes to `reports/sample/`)**
```bash
python notebooks/eda_week8.py --parquet data/processed/arxiv_parquet
```

**Full EDA (writes to `reports/full/`)**
```bash
python notebooks/eda_week8.py --parquet data/processed/arxiv_full
```

**Artifacts written** (CSV + PNG):
- Completeness by column (`completeness.csv`)
- Distinct counts for key columns (`distinct_selected.csv`)
- Text length summary (`text_length_summary.csv`)
- Top categories (`top_categories.csv/.png`)
- Papers per year (`by_year.csv/.png`)
- Category × Year heatmap (`category_year_matrix.csv/.png`)
- Abstract length histogram (`abstract_length_hist.png`, sampled)
- DOI coverage by year (`doi_rate_by_year.csv/.png`)
- Top authors (`top_authors.csv/.png`)
- Versions per paper (`version_count_hist.csv/.png`)
- Category Pareto (`category_pareto.csv/.png`)

---

## Results Snapshot (Full Run Example)
From a recent full run of the pipeline (parquet → EDA):
- **Rows after quality filters:** ~2,854,101  
- **Years covered (example top slice):** 2007–2025  
- **Median text lengths:** `title_len ≈ 72`, `abstract_len ≈ 957`  
- **DOI availability:** varies by year; see `reports/full/doi_rate_by_year.*`  
- **Top categories, authors, and more:** see CSV/PNG artifacts in `reports/full/`

> Numbers may vary across Kaggle snapshot versions and transformation filters.

---

## Make Targets (optional)
Add these to your `Makefile` for convenience:
```makefile
ingest-sample:
	python -m src.ingestion --input data/sample/arxiv-sample.jsonl --output data/processed/arxiv_parquet --partition-by year --repartition 64 --no-stats

ingest-full:
	python -m src.ingestion --input data/raw/arxiv-metadata-oai-snapshot.json --output data/processed/arxiv_full --partition-by year --repartition 200 --no-stats

eda-sample:
	python notebooks/eda_week8.py --parquet data/processed/arxiv_parquet

eda-full:
	python notebooks/eda_week8.py --parquet data/processed/arxiv_full
```

---

## Troubleshooting

### “Row count: 1” or tiny `by_year.csv`
- The Kaggle file is **JSONL**. Make sure you **do not** pass `--multiline`.
- If you did, wipe and re-run:
  ```bash
  rm -rf data/processed/arxiv_full
  python -m src.ingestion --input data/raw/arxiv-metadata-oai-snapshot.json --output data/processed/arxiv_full --partition-by year --repartition 200
  ```

### Out of Memory (OOM) during EDA
- Use the provided `run.sh` (8g driver/executor, AQE, small splits).
- Avoid `toPandas()` on large non-aggregated DataFrames.
- Reduce histogram sample: `--abslen-sample-frac 0.02` (or lower).
- Ensure local spill dir exists and has space: `data/tmp/spark-local`.

### Permissions / Java
- Ensure Java 17+ is available. In Codespaces or Ubuntu: install Temurin 17.

---

## License
- The project code is MIT (or course default).  
- Dataset metadata is CC0 (Public Domain). PDFs and individual papers may have different licenses—respect their terms.

---

## Acknowledgements
- **arXiv** (Cornell University) for maintaining the dataset and service.
- **Kaggle** for hosting a mirror and providing KaggleHub.
- **Apache Spark** community.
