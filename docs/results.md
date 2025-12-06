# Results

This document summarizes the main outcomes of the Sparxiv pipeline:

- ingestion and preprocessing statistics
- TF-IDF model configuration
- search behavior
- batch analytics
- streaming analytics
- overall interpretation

Exact numeric values depend on the specific arXiv snapshot and hardware, but the structure of the outputs and their qualitative patterns remain consistent.

---

## 1. Ingestion Results

### 1.1 Full dataset

- Raw input: `data/raw/arxiv-metadata-oai-snapshot.json`
- Output: `data/processed/arxiv_full/`
- Format: partitioned Parquet
- Partitioning: `year`
- Compression: Zstandard

Key ingestion side statistics logged during the run include:

- Total number of rows ingested, roughly matching the number of paper versions in the snapshot.
- Distinct primary categories and their counts.
- Distributional stats for `abstract_len` and `title_len`.
- Fraction of rows with missing or empty abstracts.
- DOI coverage and version count summaries via `has_doi` and `n_versions`.

### 1.2 Sample dataset

- Raw input: `data/sample/arxiv-sample.jsonl`
- Output: `data/processed/arxiv_sample/`

The sample ingestion shares the same schema and transformations as the full dataset but processes a much smaller number of rows. This makes it useful for fast iterative work and debugging.

---

## 2. TF-IDF Feature Models

Two separate models are trained, sharing the same architecture but different hyperparameters.

### 2.1 Sample model

- Training input: `data/processed/arxiv_sample/`
- Model output: `data/models/tfidf_sample/`
- Features output: `data/processed/features_sample/`

Configuration:

- Vocabulary size: 80000
- Minimum document frequency: 3
- Extra stopwords: top 200 tokens by document frequency
- Bigrams: disabled

The sample model achieves relatively dense coverage of the vocabulary used in modern abstracts and is robust against boilerplate phrasing due to the data driven stopword list.

### 2.2 Full model

- Training input: `data/processed/arxiv_full/`
- Model output: `data/models/tfidf_full/`
- Features output: `data/processed/features_full/`

Configuration:

- Vocabulary size: 120000
- Minimum document frequency: 10
- Extra stopwords: disabled
- Bigrams: disabled

The full model trades some rare term coverage for lower memory footprint and faster training. It remains expressive enough to capture meaningful terms in titles and abstracts across millions of documents.

### 2.3 Feature tables

Both feature Parquet directories expose:

- `id_base`, `paper_id`
- `title`, `abstract`, `categories`, `year`
- `features` as a Spark `VectorUDT` with L2 normalized TF-IDF values

These are used both for offline analysis and for constructing the full CSR index.

---

## 3. Search Engine Results

The search engine exposes a consistent structure for each result:

```json
{
  "rank": 1,
  "score": 0.873,
  "neighbor_id": "0704.0001",
  "paper_id": "0704.0001v1",
  "title": "Example Paper Title",
  "abstract": "We propose a method...",
  "categories": ["cs.LG"],
  "year": 2022
}
```

### 3.1 Sample mode

- All feature vectors are loaded into memory.
- Queries typically return results in well under a second.
- Top K neighbors for typical queries such as "graph neural networks" or "variational inference" are strongly on topic and often include both foundational and recent works.

This mode is ideal for demonstration during development or in limited resource environments.

### 3.2 Full mode

- The CSR index built from `features_full` is loaded once at application start.
- Query latency is dominated by a single sparse matrix vector multiplication plus top K selection.
- Results show:
  - a mix of closely related papers in the same primary category
  - thematically similar papers in adjacent or multi disciplinary categories

Because cosine similarity is purely lexical, queries are most effective when the input text resembles an arXiv style title or abstract.

---

## 4. Batch Analytics

### 4.1 Standard query outputs

Standard queries produce CSV and PNG files under:

- `reports/standard_queries_full/`
- `reports/standard_queries_sample/`

Representative outputs include:

- Papers per year, showing exponential growth in submissions and acceleration post 2010.
- Top categories and Pareto charts, with a small number of categories accounting for a large share of submissions.
- Abstract length histograms, with most abstracts in a moderate range and a long tail of very short or very long abstracts.
- Version count histograms, where most papers have 1 or 2 versions.
- DOI rate by year, where coverage improves over time.
- Top authors ranked by publication count.

### 4.2 Complex analytics

Complex analytics outputs are stored under:

- `reports/analysis_full/`
- `reports/analysis_sample/`

Key qualitative findings:

- Category co occurrence shows strong links between machine learning, computer vision, and natural language processing categories, plus connections across physics subfields.
- Author collaboration over time shows an increasing average number of authors per paper and extreme values for very large collaborations.
- Rising and declining topics highlight growth in data and learning oriented categories and plateaus or slow declines in some older subfields.
- Lexical richness and abstract length trends show modest increases in vocabulary richness and average abstract length over time.
- DOI versus versions correlation surfaces differences in revision behavior between papers with and without DOIs.
- Author lifecycle and category migration analyses reveal authors who remain in a narrow area and others who transition to newer fields.

These patterns align with known long term trends in scientific publishing.

---

## 5. Streaming Analytics

The streaming jobs write per drop reports under:

- `reports/streaming_sample/YYYYMMDD/`
- `reports/streaming_full/YYYYMMDD/`

For each date stamp, you get:

- `by_year.csv` and `papers_per_year.png` with year wise counts for that drop.
- `top_categories.csv` and `top_categories.png` with category distributions for the drop.
- `doi_rate_by_year.csv` and `doi_rate_by_year.png` with DOI coverage trends restricted to that microbatch.

Even though the underlying data is static, slicing it into simulated weekly or monthly drops allows you to validate that streaming transforms match batch transforms and to study how incremental ingestion would behave in a live system.

---

## 6. Overall Interpretation

Taken together, the results show that:

- The ingestion and transformation logic produces a clean, analytics friendly representation of the arXiv metadata.
- The TF-IDF models capture enough lexical structure to support sensible content based search on both sample and full corpora.
- The complex analytics reproduce several known trends in scientific publishing, such as growth in machine learning categories and increasing collaboration sizes.
- The streaming pipelines demonstrate how batch style metrics can be kept up to date incrementally using Structured Streaming.

While the system is not meant to be a production recommender, it provides a solid, fully reproducible baseline for:

- large scale academic text analytics
- prototype recommendation systems
- experiments with more advanced embedding based models layered on top of a Spark data pipeline.
