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

We provide both a **sample workflow** (fast, 50k records for PRs/demos) and a **full workflow** (~1.7M records). All steps are designed to run in **GitHub Codespaces** or any local Spark environment.

---

## Dataset
- **Source**: Kaggle → *Cornell-University/arxiv*  
- **Contents**: Metadata of ~1.7M arXiv papers (id, title, abstract, categories, versions, authors, DOI, etc.)  
- **Format**: JSON Lines (one record per line)  
- **Size**: ~4.5 GB (metadata JSON), growing with updates

We use `kagglehub` to download the dataset and create a small **JSONL sample** for quick iteration.

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
├─ requirements.txt
└─ README.md
```

**.gitignore** (recommended):
```
data/processed/
data/raw/
data/tmp/
.DS_Store
.ipynb_checkpoints/
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

> If needed, you can increase Spark resources by editing `src/utils.py → get_spark()` (e.g., `spark.driver.memory`, adaptive execution, shuffle partitions).

---

## 1) Download Dataset (+ Create Sample)
This downloads the full arXiv metadata and writes a 50k-line JSONL sample for quick tests.
```bash
python scripts/download_arxiv.py --sample 50000
# → raw file at data/raw/arxiv-metadata-oai-snapshot.json
# → sample JSONL at data/sample/arxiv-sample.jsonl
```

---

## 2) Ingestion (JSON/JSONL → Parquet)

### Sample (fast, for PR/demo)
```bash
python -m src.ingestion   --input data/sample/arxiv-sample.jsonl   --output data/processed/arxiv_parquet   --partition-by year
```
**Outputs**: `data/processed/arxiv_parquet/` + quick stats in console.

### Full Dataset (~1.7M records)
```bash
# IMPORTANT: Do NOT use --multiline for the Kaggle JSON (it's JSONL).
python -m src.ingestion   --input data/raw/arxiv-metadata-oai-snapshot.json   --output data/processed/arxiv_full   --partition-by year   --repartition 200
```
**Outputs**: `data/processed/arxiv_full/` (partitioned Parquet) + console stats.

> If you accidentally used `--multiline` on JSONL earlier, delete your old output with `rm -rf data/processed/arxiv_full` and re-run the command above.

---

## 3) EDA (CSV Tables + PNG Charts)
Point the EDA script at either the sample or full Parquet output.

### Sample EDA (writes to `reports/sample/`)
```bash
python notebooks/eda_week8.py --parquet data/processed/arxiv_parquet
```

### Full EDA (writes to `reports/full/`)
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

## Running in GitHub Codespaces

### Avoiding idle suspend
- Codespaces will suspend after inactivity (default 30 minutes). Increase Idle Timeout to up to 4 hours in Codespaces settings, or keep a terminal active (e.g., `watch -n 300 date`).

### tmux (optional)
Keep long jobs attached to a tmux session:
```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s eda
# run your command inside
python notebooks/eda_week8.py --parquet data/processed/arxiv_full
# detach with:  Ctrl+b then d
tmux attach -t eda   # to reattach
```

> tmux keeps your job alive if your terminal disconnects, but cannot prevent Codespaces from auto-suspending at the platform limit (e.g., 12h). For very long runs, consider chunking or a more persistent environment.

---

## Troubleshooting

### “Row count: 1” in EDA
- Likely ingestion read the raw file as a single multiline JSON object. The Kaggle file is **JSONL**; re-run ingestion **without** `--multiline`:
```bash
rm -rf data/processed/arxiv_full
python -m src.ingestion   --input data/raw/arxiv-metadata-oai-snapshot.json   --output data/processed/arxiv_full   --partition-by year   --repartition 200
```

### Out of Memory (OOM) in Codespaces
- Avoid `df.toPandas()` on large DataFrames.
- Use the **sample** for quick plots; for full runs, increase partitions and driver memory in `utils.get_spark()`.
- Reduce sampling in EDA histograms: `--abslen-sample-frac 0.02`.

### Permissions / Java
- Ensure Java 17+ is available in the container. In Codespaces, consider a devcontainer with Java + Python or install Temurin 17.

---

## License
- The project code is MIT (or course default).  
- Dataset metadata is CC0 (Public Domain). PDFs and individual papers may have different licenses—respect their terms.

---

## Acknowledgements
- **arXiv** (Cornell University) for maintaining the dataset and service.
- **Kaggle** for hosting a mirror and providing KaggleHub.
- **Apache Spark** community.
