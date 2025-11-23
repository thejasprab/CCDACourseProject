# Methodology

# 1. System Architecture Overview

## 1.1 High-Level Architecture

![High-Level Architecture](./Sparxiv_SysArchitecture_Simple.webp)

## 1.2 Detailed System Architecture

![Detailed System Architecture](./Sparxiv_SysArchitecture.webp)

---

# 2. Data Ingestion & Preprocessing (Structured APIs)

## 2.1 Ingestion Workflow
The ingestion stage loads raw, semi‑structured JSON/JSONL metadata using the Spark DataFrame API.  
This is the first step in converting arXiv’s raw export into a normalized, analyzable form.  
Spark automatically infers schema fields such as authors, abstract, categories, and timestamps, even across millions of records.  
This step ensures unified schema handling regardless of input file size.  
The ingestion layer also prepares the metadata for efficient downstream transformations.

```python
df_raw = spark.read.json(input_path)
```

## 2.2 Transformations
After loading, each record undergoes a sequence of transformations to standardize text fields and structural properties.  
Title and abstract strings are cleaned with regex-based normalization and lowercasing for consistent tokenization.  
Category strings are split into arrays, a primary category field is derived, and authors are normalized into lists.  
Temporal fields are parsed to extract the publication year, enabling partitioned storage and year‑based analytics.  
Additional derived fields—such as text lengths, author counts, and DOI flags—support both SQL analytics and ML training.

```python
df = df.withColumn("title_clean", clean_text(F.col("title")))
```

## 2.3 Storage Format
Transformations are persisted using Parquet with Zstandard compression, providing optimal read/write performance for large-scale analytics.  
Partitioning by year allows Spark SQL to efficiently prune irrelevant partitions during queries.  
Each execution of the ingestion pipeline fully overwrites the output directory to maintain reproducibility.  
This preprocessed dataset becomes the foundation for SQL analytics, ML feature generation, and the CSR index build.  
Structured storage also ensures downstream reproducibility across both sample and full pipelines.

```python
df.write.mode("overwrite").parquet(out_path)
```

---

# 3. Spark SQL Analytics

Spark SQL is used to compute a rich set of analyses on scientific publishing patterns.  
These analytics modules demonstrate Spark’s ability to perform distributed joins, aggregations, window operations, and ranking.  
Each analysis corresponds to one or more CSV/PNG outputs placed under `reports/analysis_full`.  
These results provide insight into authorship behavior, topical evolution, metadata completeness, and structural patterns within arXiv.  
Below are all analysis outputs including visualization files.

### Abstract Length vs Versions (Deciles)  
Displays distribution of abstract lengths grouped by version count.
![Abstract Length Decile](../reports/analysis_full/complex_abstractlen_versions_by_decile.png)

### Abstract Length vs Versions (Correlation)  
Examines whether longer abstracts correlate with more revisions.
![Correlation](../reports/analysis_full/complex_abstractlen_versions_correlation.png)

### Author Category Migration  
Shows how authors move between research categories over time.
![Author Category Migration](../reports/analysis_full/complex_author_category_migration_top20.png)

### Author Collaboration Over Time  
Illustrates evolving collaboration networks across years.
![Author Collaboration](../reports/analysis_full/complex_author_collab_over_time_simple.png)

### Author Lifecycle Scatter  
Plots productivity trends across author careers.
![Author Lifecycle](../reports/analysis_full/complex_author_lifecycle_scatter.png)

### Avg Token Count by Year  
Tracks shifts in linguistic complexity of abstracts.
![Avg Token Count](../reports/analysis_full/complex_avg_token_count_by_year.png)

### Category Cooccurrence  
Shows frequently co‑appearing subject categories.
![Category Cooccurrence](../reports/analysis_full/complex_category_cooccurrence_top.png)

### Category Versions Avg  
Displays average number of revisions across categories.
![Category Versions](../reports/analysis_full/complex_category_versions_avg_top30.png)

### Declining Topics  
Highlights categories decreasing in submission volume.
![Declining Topics](../reports/analysis_full/complex_declining_topics_top20.png)

### DOI Versions Correlation  
Examines whether DOI assignment correlates with revision counts.
![DOI Versions](../reports/analysis_full/complex_doi_versions_correlation.png)

### DOI vs Versions Group  
Grouped comparison of DOI presence versus version averages.
![DOI vs Versions](../reports/analysis_full/complex_doi_vs_versions_group.png)

### Lexical Richness by Year  
Measures vocabulary richness across years.
![Lexical Richness](../reports/analysis_full/complex_lexical_richness_by_year.png)

### Rising Topics  
Shows categories growing in relative prominence.
![Rising Topics](../reports/analysis_full/complex_rising_topics_top20.png)

---

# 4. Standard Queries (Full Dataset)

This section contains essential descriptive statistics computed using basic Spark SQL/DataFrame operations.  
These standard queries complement the complex analytics by summarizing global patterns in submissions, text structure, and metadata completeness.  
Each visualization is produced during the full pipeline execution and placed under `reports/standard_queries_full`.  
The figures below represent the complete set of outputs for the full dataset.  
This set also demonstrates correct usage of aggregations, grouping, windowing, and ordering.

### Abstract Length Histogram  
Distribution of abstract text lengths.
![Abstract Length Hist](../reports/standard_queries_full/abstract_length_hist.png)

### Category Pareto  
Pareto chart identifying dominant categories.
![Category Pareto](../reports/standard_queries_full/category_pareto.png)

### Category-Year Heatmap  
Heatmap showing category frequency across years.
![Category-Year Matrix](../reports/standard_queries_full/heatmap_category_year.png)

### DOI Rate by Year  
Trend in DOI usage across time.
![DOI Rate](../reports/standard_queries_full/doi_rate_by_year.png)

### Papers per Year  
Annual submission counts.
![Papers per Year](../reports/standard_queries_full/papers_per_year.png)

### Top Authors  
Top authors ranked by publication count.
![Top Authors](../reports/standard_queries_full/top_authors.png)

### Top Categories  
Most frequent arXiv primary categories.
![Top Categories](../reports/standard_queries_full/top_categories.png)

### Version Count Histogram  
Distribution of document revision counts.
![Version Count](../reports/standard_queries_full/version_count_hist.png)

---

# 5. ML Pipeline

The ML pipeline uses Spark MLlib to generate TF‑IDF features from cleaned text.  
This includes tokenization, stopword removal, vocabulary construction, TF computation, IDF weighting, and L2 normalization.  
These stages convert raw scientific text into high‑dimensional sparse vectors suitable for similarity search.  
The same pipeline is reused for both sample and full datasets to ensure vector consistency.  
This trained pipeline is also applied during query-time vectorization in the Flask application.

```python
pipeline = Pipeline(stages=[tokenizer, stopwords, vectorizer, idf, normalizer])
```

---

# 6. Similarity Search

Similarity search is implemented in two modes to accommodate dataset scale.  
Sample mode uses Python in‑memory SparseVectors, enabling fast dot‑product scoring.  
Full mode uses a prebuilt CSR index, created offline from the full TF‑IDF feature set.  
This avoids Spark cross‑joins at query time, ensuring low latency even for millions of documents.  
Both modes rely on the Spark ML pipeline to ensure consistent query vector representation.

### Sample Mode  
```python
score = dot(q.toArray(), v.toArray())
```

### Full Mode (CSR Index)  
```python
scores = csr_matrix @ query_dense
```

---

# 7. Streaming

The streaming workflow processes simulated real‑time updates to the arXiv dataset.  
Spark Structured Streaming continuously monitors a directory for new JSONL “drops.”  
Each microbatch is transformed using the same preprocessing logic as the batch pipeline.  
Yearly counts, category summaries, and DOI statistics are recomputed incrementally.  
This demonstrates Spark’s ability to unify batch and streaming computations under a shared codebase.

```python
stream_df = spark.readStream.schema(schema).json(stream_path)
```
