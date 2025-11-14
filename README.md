# ITCS 6190/8190 – Cloud Computing for Data Analysis
## Course Project: Data Analysis with Apache Spark (arXiv on Kaggle)

### Team Members
1. **Ali Khaleghi Rahimian** — akhalegh@charlotte.edu  
2. **Kiyoung Kim** — kkim43@charlotte.edu  
3. **Thejas Prabakaran** — tprabaka@charlotte.edu

---

## Overview
This project implements a **Spark-based data analysis pipeline** for the **Cornell arXiv metadata** dataset (via Kaggle). We cover:

- **Data ingestion** (JSON/JSONL → cleaned & partitioned Parquet)  
- **Transformations** (text cleanup, field standardization, quality filters)  
- **Exploratory Data Analysis (EDA)** (counts, trends, categories, DOI coverage, etc.)  
- **Streaming (Week 11)**: structured streaming of new snapshots with **per-drop reports**  
- **ML Recommender (Week 12)**: **content-based retrieval** using **TF-IDF + cosine** with metrics (Precision@K, Recall@K, MAP@K, MRR, coverage) and ad-hoc **query** support

We provide both a **sample workflow** (fast, ~50k records) and a **full workflow** (≈1.7M–2.8M+ records after filters). Everything runs in **local Spark** or **Codespaces**.

---

## Dataset
- **Source**: Kaggle → *Cornell-University/arxiv*  
- **Format**: JSON Lines (one record per line, **JSONL**)  
- **Size**: ~4–6 GB (metadata), grows with updates
- **Raw path** (expected): `data/raw/arxiv-metadata-oai-snapshot.json`

---

## Repository Structure
```
.
├─ scripts/
│  ├─ download_arxiv.py                  # Download full Kaggle dataset (+ optional sample)
│  ├─ prepare_sample_stream_batches.py   # Week 11: generate weekly sample drops
│  ├─ split_sample.py                    # Week 12 (sample) split helper
│  ├─ split.py                           # Week 12 (full) split helper
│  ├─ train_ml_sample.py                 # Week 12 (sample) train TF-IDF
│  └─ train_ml.py                        # Week 12 (full) train TF-IDF
├─ notebooks/
│  ├─ eda.py                       # EDA for sample/full
│  ├─ streaming_sample.py         # Week 11 streaming (sample)
│  ├─ streaming.py                # Week 11 streaming (full weekly)
│  ├─ ml_sample.py                # Week 12 (sample) eval/query
│  └─ ml.py                       # Week 12 (full) eval/query
├─ src/
│  ├─ ingestion.py
│  ├─ transformations.py
│  ├─ utils.py
│  ├─ featurization.py                   # Tokenizer/stopwords/TF-IDF + L2
│  ├─ similarity.py                      # Cosine + Top-K helpers
│  └─ query.py                           # Vectorize free-text query
├─ data/
│  ├─ raw/                               # Raw JSONL (ignored by Git)
│  ├─ sample/                            # Tiny JSONL sample (ignored by Git)
│  └─ processed/                         # Parquet outputs (ignored by Git)
├─ reports/
│  ├─ sample/                            # EDA outputs (sample)
│  ├─ full/                              # EDA outputs (full)
│  ├─ streaming_sample/                  # Week 11 sample streaming reports
│  └─ streaming_full/                    # Week 11 full weekly reports
├─ run.sh                                # Full ingestion + EDA
├─ run_sample.sh                         # Sample ingestion + EDA
├─ run_ml_sample.sh                      # Week 12 SAMPLE: split/train/eval
├─ run_ml.sh                             # Week 12 FULL: split/train/eval
├─ requirements.txt
└─ README.md
```

---

## Environment & Requirements
Install Python deps:
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

> **Java**: Use Java 17+ (e.g., Temurin 17).  
> **Memory**: You can raise Spark memory via env vars:
> ```bash
> export SPARK_DRIVER_MEMORY=10g
> export SPARK_EXECUTOR_MEMORY=10g
> ```

---

## Quick Start (Weeks 8–9: Ingestion + EDA)

### A) Full Pipeline (ingestion → EDA)
```bash
bash run.sh
```
Writes full EDA artifacts under `reports/full/`.

### B) Sample Pipeline (ingestion → EDA)
```bash
bash run_sample.sh
```
Writes sample EDA artifacts under `reports/sample/`.

**Manual ingestion commands**

Sample (fast):
```bash
python -m src.ingestion   --input data/sample/arxiv-sample.jsonl   --output data/processed/arxiv_parquet   --partition-by year   --repartition 64
```

Full (JSONL; **do not** use `--multiline`):
```bash
python -m src.ingestion   --input data/raw/arxiv-metadata-oai-snapshot.json   --output data/processed/arxiv_full   --partition-by year   --repartition 200
```

EDA (sample → `reports/sample/`, full → `reports/full/`):
```bash
python notebooks/eda.py --parquet data/processed/arxiv_parquet
python notebooks/eda.py --parquet data/processed/arxiv_full
```

---

## Week 11 — Streaming (recap)
See original README details. Sample streaming emits weekly-dated drops and regenerates reports; the full runner pulls Kaggle on Sundays. Paths:
- Incoming (sample): `data/stream/incoming_sample/`
- Incoming (full): `data/stream/incoming/`
- Reports: `reports/streaming_sample/YYYYMMDD/`, `reports/streaming_full/YYYYMMDD/`

---

## Week 12 — ML Recommender (TF-IDF + Cosine)

### What it does
- Trains a **TF-IDF** model on `title + abstract` (with default + domain + top-DF extra stopwords).  
- Produces **L2-normalized vectors** (`features`) for **train/test** splits.  
- **Evaluation** (Top-K retrieval): computes **Precision@K**, **Recall@K**, **MAP@K**, **MRR**, **coverage**.  
- **Query mode**: turns an arbitrary `title + abstract` into a vector and returns Top-K neighbors from TRAIN.

### Artifacts
- **Model**: `data/models/tfidf_*` (`model.json` contains metadata)  
- **Features**: `data/processed/features_trained_*/split={train,test}`  
- **Reports**:
  - `recs_topk.csv` — Top-K neighbors per test doc
  - `metrics_at_k.csv` — macro metrics
  - `qualitative_examples.md` — quick inspection list

---

## ML (Sample Dataset)

> Uses existing sample parquet at `data/processed/arxiv_parquet` (from ingestion of `data/sample/arxiv-sample.jsonl`).

### 1) Split (sample)
```bash
python scripts/split_sample.py   --parquet data/processed/arxiv_parquet   --out data/processed/arxiv_sample_split   --test-years 2007,2008,2009   --test-size 1000   --seed 42
```

### 2) Train TF-IDF (sample)
```bash
python scripts/train_ml_sample.py   --split-parquet data/processed/arxiv_sample_split   --model-dir data/models/tfidf_sample   --features-out data/processed/features_trained_sample   --vocab-size 80000   --min-df 3   --use-bigrams false   --extra-stopwords-topdf 200   --seed 42
```

### 3) Evaluate (sample)
```bash
python notebooks/ml_sample.py   --mode eval   --model-dir data/models/tfidf_sample   --split-parquet data/processed/arxiv_sample_split   --features data/processed/features_trained_sample   --out reports/ml_sample   --k 3   --strategy exact
```

### 4) Query (sample; optional)
```bash
python notebooks/ml_sample.py   --mode query   --model-dir data/models/tfidf_sample   --features-train data/processed/features_trained_sample/split=train   --out reports/ml_sample   --query-title "Graph Neural Networks for Molecules"   --query-abstract "We propose a message passing architecture ..."   --k 5
```

### One-command runner (sample)
```bash
bash run_ml_sample.sh
```

---

## ML (Full Dataset)

> Assumes you have full Parquet at `data/processed/arxiv_full` (see ingestion section).

### 1) Split (full)
```bash
python scripts/split.py   --parquet data/processed/arxiv_full   --out data/processed/arxiv_split   --test-years 2019,2020,2021   --test-size 50000   --seed 42
```

### 2) Train TF-IDF (full)
```bash
python scripts/train_ml.py   --split-parquet data/processed/arxiv_split   --model-dir data/models/tfidf_full   --features-out data/processed/features_trained_full   --vocab-size 250000   --min-df 5   --use-bigrams false   --extra-stopwords-topdf 500   --seed 42
```

### 3) Evaluate (full)
**Default strategy (`block_cat`) scales to the full set** by only comparing items within shared categories (proxy for recall):  
```bash
python notebooks/ml.py   --mode eval   --model-dir data/models/tfidf_full   --split-parquet data/processed/arxiv_split   --features data/processed/features_trained_full   --out reports/full   --k 5   --strategy block_cat   --eval-max-test 20000
```
> `--strategy exact_broadcast` is only for very small training sets; do **not** use on full.

### 4) Query (full; optional)
```bash
python notebooks/ml.py   --mode query   --model-dir data/models/tfidf_full   --features-train data/processed/features_trained_full/split=train   --out reports/full   --query-title "Diffusion models for conditional generation"   --query-abstract "We introduce a guidance method ..."   --k 10
```

### One-command runner (full)
```bash
bash run_ml.sh
```

---

## Configuration Knobs (ML)
- **Vocabulary**: `--vocab-size` (sample: 80k, full: 250k recommended)  
- **Min DF**: `--min-df` (sample: 3, full: 5)  
- **Extra stopwords from top DF**: `--extra-stopwords-topdf` (sample: 200, full: 500)  
- **Bigrams**: `--use-bigrams true|false` (default false; increases feature size)  
- **Evaluation K**: `--k` (sample default 3; full default 5–10)  
- **Eval sample size**: `--eval-max-test` to bound compute on full test set  
- **Strategy**: `exact` (sample) vs `block_cat` (full)

---

## Results Snapshot (Full ML)
- **Artifacts** in `reports/full/`:
  - `recs_topk.csv`, `metrics_at_k.csv`, `qualitative_examples.md`
- Metrics vary by snapshot and knobs; reported values include **Precision@K**, **Recall@K**, **MAP@K**, **MRR**, and **coverage**.

---

## Troubleshooting (ML)

**IllegalArgumentException: Path from empty string**  
You likely passed an **empty env var** in `--parquet` or `--out`. Use literal paths or `export VAR=value` first.

**OOM / slow shuffles on full**  
- Increase memory (`SPARK_DRIVER_MEMORY`, `SPARK_EXECUTOR_MEMORY`).  
- Ensure local spill dir exists (e.g., `data/tmp/spark-local`) and has space.  
- Keep `block_cat` for eval; avoid full cartesian joins.  
- Lower `--eval-max-test` if needed.

**“JSON vs JSONL” confusion**  
Kaggle’s file is **JSONL**. Do **not** pass `--multiline` to ingestion.

**Hostname / native-hadoop WARNs**  
Harmless. To quiet hostname warning:
```bash
export SPARK_LOCAL_IP=127.0.0.1   # or your interface IP
```

---

## Make Targets (optional)
```makefile
# Ingestion
ingest-sample:
	python -m src.ingestion --input data/sample/arxiv-sample.jsonl --output data/processed/arxiv_parquet --partition-by year --repartition 64 --no-stats

ingest-full:
	python -m src.ingestion --input data/raw/arxiv-metadata-oai-snapshot.json --output data/processed/arxiv_full --partition-by year --repartition 200 --no-stats

# EDA
eda-sample:
	python notebooks/eda.py --parquet data/processed/arxiv_parquet

eda-full:
	python notebooks/eda.py --parquet data/processed/arxiv_full

# Week 12 ML — Sample
ml-split-sample:
	python scripts/split_sample.py --parquet data/processed/arxiv_parquet --out data/processed/arxiv_sample_split --test-years 2007,2008,2009 --test-size 1000 --seed 42

ml-train-sample:
	python scripts/train_ml_sample.py --split-parquet data/processed/arxiv_sample_split --model-dir data/models/tfidf_sample --features-out data/processed/features_trained_sample --vocab-size 80000 --min-df 3 --use-bigrams false --extra-stopwords-topdf 200 --seed 42

ml-eval-sample:
	python notebooks/ml_sample.py --mode eval --model-dir data/models/tfidf_sample --split-parquet data/processed/arxiv_sample_split --features data/processed/features_trained_sample --out reports/ml_sample --k 3 --strategy exact

# Week 12 ML — Full
ml-split-full:
	python scripts/split.py --parquet data/processed/arxiv_full --out data/processed/arxiv_split --test-years 2019,2020,2021 --test-size 50000 --seed 42

ml-train-full:
	python scripts/train_ml.py --split-parquet data/processed/arxiv_split --model-dir data/models/tfidf_full --features-out data/processed/features_trained_full --vocab-size 250000 --min-df 5 --use-bigrams false --extra-stopwords-topdf 500 --seed 42

ml-eval-full:
	python notebooks/ml.py --mode eval --model-dir data/models/tfidf_full --split-parquet data/processed/arxiv_split --features data/processed/features_trained_full --out reports/full --k 5 --strategy block_cat --eval-max-test 20000
```
---

## License
- Code: MIT (or course default).  
- Dataset metadata: CC0 (Public Domain). PDFs and individual papers may carry different licenses.

---

## Acknowledgements
- **arXiv** (Cornell University) for maintaining the dataset and service  
- **Kaggle** for hosting the mirror and providing KaggleHub  
- **Apache Spark** community

