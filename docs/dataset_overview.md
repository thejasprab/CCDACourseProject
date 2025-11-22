# Dataset Overview

This project is built on top of the **arXiv Dataset** provided by *Cornell University* and hosted on **Kaggle**.  
The dataset contains metadata for **1.7M–2.8M+ scholarly papers** across physics, computer science, mathematics, statistics, electrical engineering, quantitative biology, economics, and related STEM fields.

Dataset link:  
➡️ https://www.kaggle.com/datasets/Cornell-University/arxiv

---

## 1. About arXiv

arXiv is one of the oldest and most influential open-access repositories of scientific articles, founded by Paul Ginsparg in 1991 and maintained by Cornell University.  
For over 30 years, it has provided free access to cutting-edge scientific research across a wide range of domains—particularly physics, mathematics, computer science, and quantitative fields.

The repository contains:

- Preprints and published versions  
- Multiple versions per paper  
- Rich metadata and structured categories  
- High-quality abstracts and author information  

---

## 2. About the Kaggle Dataset

The Kaggle dataset is a **machine-readable mirror** of arXiv’s metadata, designed for research in:

- Trend analysis  
- Paper recommendation systems  
- Category prediction  
- Co-citation and collaboration network analysis  
- Knowledge graph construction  
- Semantic search interfaces  
- Large-scale model training or fine-tuning  

It was developed through a collaboration between Cornell University and several contributors.

**Kaggle Dataset Page:**  
https://www.kaggle.com/datasets/Cornell-University/arxiv

---

## 3. Dataset Contents

The dataset contains a single JSON Lines file:

- **File:** `arxiv-metadata-oai-snapshot.json`  
- **Size:** ~4.96 GB (metadata only)  
- **Format:** JSONL (one paper per line)

Since the **full arXiv dataset is 1.1TB and continually growing**, the Kaggle version only includes metadata rather than PDFs.

Each entry contains the following fields:

| Field | Description |
|-------|-------------|
| `id` | arXiv identifier (e.g., 0704.0001) |
| `submitter` | Name of the submitter |
| `authors` | Raw author string |
| `title` | Paper title |
| `comments` | Additional notes (pages, figures, etc.) |
| `journal-ref` | Journal publication info |
| `doi` | DOI if available |
| `report-no` | Technical report identifier |
| `categories` | Space-separated list of arXiv categories |
| `abstract` | Full abstract text (often contains LaTeX) |
| `versions` | List of submission versions |
| `update_date` | Last update timestamp |
| `authors_parsed` | Structured list of author fields |

---

## 4. Access to Full PDFs (Google Cloud Storage)

While the Kaggle dataset contains only metadata, the full text PDFs are freely accessible via Google Cloud Storage:

```
gs://arxiv-dataset/
```

Example (using gsutil):

```bash
# List files:
gsutil cp gs://arxiv-dataset/arxiv/

# Download PDFs from March 2020:
gsutil cp gs://arxiv-dataset/arxiv/arxiv/pdf/2003/ ./local_dir/

# Download all source files:
gsutil cp -r gs://arxiv-dataset/arxiv/ ./local_dir/
```

These can be used to extend Sparxiv into a full-text semantic search system.

---

## 5. Update Frequency

- Metadata and GCS buckets are updated **weekly**  
- Kaggle dataset shows **expected update frequency: Monthly**  

Sparxiv does not automatically ingest live updates, but the streaming module simulates weekly ingestion using sample slices.

---

## 6. Dataset Statistics

### 6.1 Size & Growth
- 1.7M+ papers  
- Continuous growth since 1990s  
- Strong acceleration after 2010  
- Dramatic expansion in cs.LG, cs.CL, stat.ML

### 6.2 Category Distribution
Includes over **150+ arXiv categories**, covering:

- Physics (astro-ph, hep-ph, cond-mat, quant-ph)  
- Math (math.AG, math.NT, math.PR)  
- Computer Science (cs.LG, cs.CL, cs.AI, cs.CV)  
- Statistics, Economics, Quantitative Biology  

Highly imbalanced distribution impacts analytics and model behavior.

### 6.3 Metadata Characteristics
- Multi-category labels  
- Multiple submission versions per paper  
- Missing fields (DOI, comments, journal-ref)  
- LaTeX-heavy abstracts  
- Wide variation in text length (20–4000+ chars)

---

## 7. How Sparxiv Uses the Dataset

### 7.1 Batch Ingestion (JSONL → Parquet)
Processes raw metadata:

- Normalize whitespace; abstracts may contain LaTeX but are used as-is for TF-IDF (no full LaTeX stripping)
- Standardizes timestamps by extracting publication year from update_date
- Extract primary categories  
- Filter incomplete abstracts  
- Write Parquet partitioned by year/category

### 7.2 Feature Engineering
Using Spark ML:

- Tokenization  
- Stopword filtering  
- TF-IDF transformation  
- L2 normalization  
- Produce sparse vector features

### 7.3 Complex Analytics
Using Spark SQL:

- Category co-occurrence  
- DOI completeness  
- Author collaboration trends  
- Temporal topic analysis  
- Abstract readability & length trends  

### 7.4 Recommendation Engine
Uses:

- Title + abstract text  
- TF-IDF vectors  
- Cosine similarity search (Top-K)  
- Query vectorization for free-text search  

### 7.5 Streaming Simulation
Breaks sample dataset into weekly JSONL drops:

- Mimics real arXiv weekly updates  
- Produces per-batch analytics (CSV + plots)

---

## 8. Summary

The arXiv dataset is a rich, large-scale scientific corpus that provides:

- Multi-domain scholarly metadata  
- Long-term temporal coverage  
- Category-structured labels  
- High-quality abstracts ideal for TF-IDF modeling  
- A strong foundation for analytics, streaming, and recommendation systems

Sparxiv leverages this dataset to demonstrate:

- Scalable data ingestion  
- High-volume feature engineering  
- Content-based recommendation  
- Temporal and category analytics  
- Real-time streaming simulations

This makes arXiv an ideal dataset for both data engineering and machine learning pipelines.
