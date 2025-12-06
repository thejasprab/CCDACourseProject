# Reports Overview

This document catalogs all reports produced by the Sparxiv pipeline:

- Standard query summaries for the full and sample datasets
- Complex analytics for both modes
- Streaming analytics snapshots for multiple dates
- Corresponding CSV tables and PNG figures wherever available

All paths are relative to the project root. Image links here assume this file lives in `docs/` and refer to images under `../reports/...`.

---

## 1. Standard Queries

Standard queries provide core descriptive statistics on the arXiv metadata. They are computed for both the full dataset and a smaller sample.

### 1.1 Full dataset

Directory:

```text
reports/standard_queries_full/
```

CSV files:

- `by_year.csv`
- `category_pareto.csv`
- `category_year_matrix.csv`
- `completeness.csv`
- `distinct_selected.csv`
- `doi_rate_by_year.csv`
- `text_length_summary.csv`
- `top_authors.csv`
- `top_categories.csv`
- `version_count_hist.csv`

Available figures:

#### Papers per year

- CSV: `reports/standard_queries_full/by_year.csv`
- Figure:  

  ![Papers per year (full)](../reports/standard_queries_full/papers_per_year.png)

#### Category year matrix

- CSV: `reports/standard_queries_full/category_year_matrix.csv`
- Figure (heatmap):  

  ![Category year heatmap (full)](../reports/standard_queries_full/heatmap_category_year.png)

#### Top categories

- CSV: `reports/standard_queries_full/top_categories.csv`
- Figure:  

  ![Top categories (full)](../reports/standard_queries_full/top_categories.png)

#### Category Pareto

- CSV: `reports/standard_queries_full/category_pareto.csv`
- Figure:  

  ![Category Pareto (full)](../reports/standard_queries_full/category_pareto.png)

#### DOI rate by year

- CSV: `reports/standard_queries_full/doi_rate_by_year.csv`
- Figure:  

  ![DOI rate by year (full)](../reports/standard_queries_full/doi_rate_by_year.png)

#### Text length summary

- CSV: `reports/standard_queries_full/text_length_summary.csv`
- Abstract length histogram figure:  

  ![Abstract length histogram (full)](../reports/standard_queries_full/abstract_length_hist.png)

#### Top authors

- CSV: `reports/standard_queries_full/top_authors.csv`
- Figure:  

  ![Top authors (full)](../reports/standard_queries_full/top_authors.png)

#### Version count histogram

- CSV: `reports/standard_queries_full/version_count_hist.csv`
- Figure:  

  ![Version count histogram (full)](../reports/standard_queries_full/version_count_hist.png)

#### Completeness

- CSV: `reports/standard_queries_full/completeness.csv`
- No dedicated figure is referenced; data is intended for tabular inspection.

#### Distinct selected

- CSV: `reports/standard_queries_full/distinct_selected.csv`
- No dedicated figure is referenced; this is a supporting table.

---

### 1.2 Sample dataset

Directory:

```text
reports/standard_queries_sample/
```

CSV files:

- `by_year.csv`
- `category_pareto.csv`
- `category_year_matrix.csv`
- `completeness.csv`
- `distinct_selected.csv`
- `doi_rate_by_year.csv`
- `text_length_summary.csv`
- `top_authors.csv`
- `top_categories.csv`
- `version_count_hist.csv`

Figures (same metrics as full, on the sample subset):

#### Papers per year

- CSV: `reports/standard_queries_sample/by_year.csv`
- Figure:  

  ![Papers per year (sample)](../reports/standard_queries_sample/papers_per_year.png)

#### Category year matrix

- CSV: `reports/standard_queries_sample/category_year_matrix.csv`
- Figure (heatmap):  

  ![Category year heatmap (sample)](../reports/standard_queries_sample/heatmap_category_year.png)

#### Top categories

- CSV: `reports/standard_queries_sample/top_categories.csv`
- Figure:  

  ![Top categories (sample)](../reports/standard_queries_sample/top_categories.png)

#### Category Pareto

- CSV: `reports/standard_queries_sample/category_pareto.csv`
- Figure:  

  ![Category Pareto (sample)](../reports/standard_queries_sample/category_pareto.png)

#### DOI rate by year

- CSV: `reports/standard_queries_sample/doi_rate_by_year.csv`
- Figure:  

  ![DOI rate by year (sample)](../reports/standard_queries_sample/doi_rate_by_year.png)

#### Text length summary

- CSV: `reports/standard_queries_sample/text_length_summary.csv`
- Abstract length histogram figure:  

  ![Abstract length histogram (sample)](../reports/standard_queries_sample/abstract_length_hist.png)

#### Top authors

- CSV: `reports/standard_queries_sample/top_authors.csv`
- Figure:  

  ![Top authors (sample)](../reports/standard_queries_sample/top_authors.png)

#### Version count histogram

- CSV: `reports/standard_queries_sample/version_count_hist.csv`
- Figure:  

  ![Version count histogram (sample)](../reports/standard_queries_sample/version_count_hist.png)

#### Completeness

- CSV: `reports/standard_queries_sample/completeness.csv`
- No dedicated figure is referenced.

#### Distinct selected

- CSV: `reports/standard_queries_sample/distinct_selected.csv`
- No dedicated figure is referenced.

---

## 2. Complex Analytics

Complex analytics capture higher level temporal, lexical, and author level patterns. They are computed for both full and sample datasets.

### 2.1 Full dataset

Directory:

```text
reports/analysis_full/
```

CSV files:

- `complex_abstractlen_versions_by_decile.csv`
- `complex_abstractlen_versions_correlation.csv`
- `complex_author_category_migration.csv`
- `complex_author_collab_over_time_simple.csv`
- `complex_author_lifecycle_top.csv`
- `complex_category_cooccurrence.csv`
- `complex_category_versions_avg.csv`
- `complex_declining_topics_top20.csv`
- `complex_doi_versions_correlation.csv`
- `complex_doi_vs_versions_group.csv`
- `complex_lexical_richness_by_year.csv`
- `complex_rising_declining_topics_fullrank.csv`
- `complex_rising_topics_top20.csv`

Available figures:

#### Abstract length vs versions by decile

- CSV: `reports/analysis_full/complex_abstractlen_versions_by_decile.csv`
- Figure:  

  ![Abstract length vs versions by decile (full)](../reports/analysis_full/complex_abstractlen_versions_by_decile.png)

#### Abstract length vs versions correlation

- CSV: `reports/analysis_full/complex_abstractlen_versions_correlation.csv`
- Figure:  

  ![Abstract length vs versions correlation (full)](../reports/analysis_full/complex_abstractlen_versions_correlation.png)

#### Author category migration

- CSV: `reports/analysis_full/complex_author_category_migration.csv`
- Figure (top categories):  

  ![Author category migration (full)](../reports/analysis_full/complex_author_category_migration_top20.png)

#### Author collaboration over time

- CSV: `reports/analysis_full/complex_author_collab_over_time_simple.csv`
- Figure:  

  ![Author collaboration over time (full)](../reports/analysis_full/complex_author_collab_over_time_simple.png)

#### Author lifecycle

- CSV: `reports/analysis_full/complex_author_lifecycle_top.csv`
- Figure (scatter):  

  ![Author lifecycle scatter (full)](../reports/analysis_full/complex_author_lifecycle_scatter.png)

#### Average token count by year

- CSV: `reports/analysis_full/complex_lexical_richness_by_year.csv`
- Figure:  

  ![Average token count by year (full)](../reports/analysis_full/complex_avg_token_count_by_year.png)

#### Lexical richness by year

- CSV: `reports/analysis_full/complex_lexical_richness_by_year.csv`
- Figure:  

  ![Lexical richness by year (full)](../reports/analysis_full/complex_lexical_richness_by_year.png)

#### Category cooccurrence

- CSV: `reports/analysis_full/complex_category_cooccurrence.csv`
- Figure (top pairs):  

  ![Category cooccurrence (full)](../reports/analysis_full/complex_category_cooccurrence_top.png)

#### Category versions average

- CSV: `reports/analysis_full/complex_category_versions_avg.csv`
- Figure (top categories):  

  ![Category versions average (full)](../reports/analysis_full/complex_category_versions_avg_top30.png)

#### Rising topics

- CSV: `reports/analysis_full/complex_rising_topics_top20.csv`
- Figure:  

  ![Rising topics (full)](../reports/analysis_full/complex_rising_topics_top20.png)

#### Declining topics

- CSV: `reports/analysis_full/complex_declining_topics_top20.csv`
- Figure:  

  ![Declining topics (full)](../reports/analysis_full/complex_declining_topics_top20.png)

#### DOI vs versions correlation

- CSV: `reports/analysis_full/complex_doi_versions_correlation.csv`
- Figure:  

  ![DOI vs versions correlation (full)](../reports/analysis_full/complex_doi_versions_correlation.png)

#### DOI vs versions group comparison

- CSV: `reports/analysis_full/complex_doi_vs_versions_group.csv`
- Figure:  

  ![DOI vs versions grouped (full)](../reports/analysis_full/complex_doi_vs_versions_group.png)

*(The full ranking across all categories is contained in `complex_rising_declining_topics_fullrank.csv`.)*

---

### 2.2 Sample dataset

Directory:

```text
reports/analysis_sample/
```

CSV files:

- `complex_abstractlen_versions_by_decile.csv`
- `complex_abstractlen_versions_correlation.csv`
- `complex_author_category_migration.csv`
- `complex_author_collab_over_time_simple.csv`
- `complex_author_lifecycle_top.csv`
- `complex_author_pairs_by_year.csv`
- `complex_category_cooccurrence.csv`
- `complex_category_versions_avg.csv`
- `complex_declining_topics_top20.csv`
- `complex_doi_versions_correlation.csv`
- `complex_doi_vs_versions_group.csv`
- `complex_lexical_richness_by_year.csv`
- `complex_rising_declining_topics_fullrank.csv`
- `complex_rising_topics_top20.csv`

Figures:

#### Abstract length vs versions by decile

- CSV: `reports/analysis_sample/complex_abstractlen_versions_by_decile.csv`
- Figure:  

  ![Abstract length vs versions by decile (sample)](../reports/analysis_sample/complex_abstractlen_versions_by_decile.png)

#### Abstract length vs versions correlation

- CSV: `reports/analysis_sample/complex_abstractlen_versions_correlation.csv`
- Figure:  

  ![Abstract length vs versions correlation (sample)](../reports/analysis_sample/complex_abstractlen_versions_correlation.png)

#### Author category migration

- CSV: `reports/analysis_sample/complex_author_category_migration.csv`
- Figure:  

  ![Author category migration (sample)](../reports/analysis_sample/complex_author_category_migration_top20.png)

#### Author collaboration over time

- CSV: `reports/analysis_sample/complex_author_collab_over_time_simple.csv`
- Figure:  

  ![Author collaboration over time (sample)](../reports/analysis_sample/complex_author_collab_over_time_simple.png)

#### Author lifecycle

- CSV: `reports/analysis_sample/complex_author_lifecycle_top.csv`
- Figure:  

  ![Author lifecycle scatter (sample)](../reports/analysis_sample/complex_author_lifecycle_scatter.png)

#### Author pairs by year

- CSV: `reports/analysis_sample/complex_author_pairs_by_year.csv`
- Figure (if generated):  

  ![Author pairs by year (sample)](../reports/analysis_sample/complex_author_pairs_by_year.png)

#### Average token count by year

- CSV: `reports/analysis_sample/complex_lexical_richness_by_year.csv`
- Figure:  

  ![Average token count by year (sample)](../reports/analysis_sample/complex_avg_token_count_by_year.png)

#### Lexical richness by year

- CSV: `reports/analysis_sample/complex_lexical_richness_by_year.csv`
- Figure:  

  ![Lexical richness by year (sample)](../reports/analysis_sample/complex_lexical_richness_by_year.png)

#### Category cooccurrence

- CSV: `reports/analysis_sample/complex_category_cooccurrence.csv`
- Figure:  

  ![Category cooccurrence (sample)](../reports/analysis_sample/complex_category_cooccurrence_top.png)

#### Category versions average

- CSV: `reports/analysis_sample/complex_category_versions_avg.csv`
- Figure:  

  ![Category versions average (sample)](../reports/analysis_sample/complex_category_versions_avg_top30.png)

#### Rising topics

- CSV: `reports/analysis_sample/complex_rising_topics_top20.csv`
- Figure:  

  ![Rising topics (sample)](../reports/analysis_sample/complex_rising_topics_top20.png)

#### Declining topics

- CSV: `reports/analysis_sample/complex_declining_topics_top20.csv`
- Figure:  

  ![Declining topics (sample)](../reports/analysis_sample/complex_declining_topics_top20.png)

#### DOI vs versions correlation

- CSV: `reports/analysis_sample/complex_doi_versions_correlation.csv`
- Figure:  

  ![DOI vs versions correlation (sample)](../reports/analysis_sample/complex_doi_versions_correlation.png)

#### DOI vs versions group comparison

- CSV: `reports/analysis_sample/complex_doi_vs_versions_group.csv`
- Figure:  

  ![DOI vs versions grouped (sample)](../reports/analysis_sample/complex_doi_vs_versions_group.png)

---

## 3. Streaming Analytics

Streaming analytics simulate periodic arXiv metadata drops and compute the same metrics per microbatch. Reports are written per date stamp.

Each streaming date directory contains:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`
- `papers_per_year.png`
- `doi_rate_by_year.png`
- `top_categories.png`

### 3.1 Full dataset streaming

Base directory:

```text
reports/streaming_full/
```

Available dates:

- `20251026/`
- `20251102/`
- `20251109/`
- `20251116/`

#### 3.1.1 Date 2025 10 26

Directory:

```text
reports/streaming_full/20251026/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 10 26 (full streaming)](../reports/streaming_full/20251026/papers_per_year.png)
- ![DOI rate by year, 2025 10 26 (full streaming)](../reports/streaming_full/20251026/doi_rate_by_year.png)
- ![Top categories, 2025 10 26 (full streaming)](../reports/streaming_full/20251026/top_categories.png)

#### 3.1.2 Date 2025 11 02

Directory:

```text
reports/streaming_full/20251102/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 11 02 (full streaming)](../reports/streaming_full/20251102/papers_per_year.png)
- ![DOI rate by year, 2025 11 02 (full streaming)](../reports/streaming_full/20251102/doi_rate_by_year.png)
- ![Top categories, 2025 11 02 (full streaming)](../reports/streaming_full/20251102/top_categories.png)

#### 3.1.3 Date 2025 11 09

Directory:

```text
reports/streaming_full/20251109/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 11 09 (full streaming)](../reports/streaming_full/20251109/papers_per_year.png)
- ![DOI rate by year, 2025 11 09 (full streaming)](../reports/streaming_full/20251109/doi_rate_by_year.png)
- ![Top categories, 2025 11 09 (full streaming)](../reports/streaming_full/20251109/top_categories.png)

#### 3.1.4 Date 2025 11 16

Directory:

```text
reports/streaming_full/20251116/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 11 16 (full streaming)](../reports/streaming_full/20251116/papers_per_year.png)
- ![DOI rate by year, 2025 11 16 (full streaming)](../reports/streaming_full/20251116/doi_rate_by_year.png)
- ![Top categories, 2025 11 16 (full streaming)](../reports/streaming_full/20251116/top_categories.png)

---

### 3.2 Sample dataset streaming

Base directory:

```text
reports/streaming_sample/
```

Available dates:

- `20251114/`
- `20251121/`
- `20251128/`
- `20251205/`
- `20251212/`

Each date directory has the same structure.

#### 3.2.1 Date 2025 11 14

Directory:

```text
reports/streaming_sample/20251114/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 11 14 (sample streaming)](../reports/streaming_sample/20251114/papers_per_year.png)
- ![DOI rate by year, 2025 11 14 (sample streaming)](../reports/streaming_sample/20251114/doi_rate_by_year.png)
- ![Top categories, 2025 11 14 (sample streaming)](../reports/streaming_sample/20251114/top_categories.png)

#### 3.2.2 Date 2025 11 21

Directory:

```text
reports/streaming_sample/20251121/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 11 21 (sample streaming)](../reports/streaming_sample/20251121/papers_per_year.png)
- ![DOI rate by year, 2025 11 21 (sample streaming)](../reports/streaming_sample/20251121/doi_rate_by_year.png)
- ![Top categories, 2025 11 21 (sample streaming)](../reports/streaming_sample/20251121/top_categories.png)

#### 3.2.3 Date 2025 11 28

Directory:

```text
reports/streaming_sample/20251128/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 11 28 (sample streaming)](../reports/streaming_sample/20251128/papers_per_year.png)
- ![DOI rate by year, 2025 11 28 (sample streaming)](../reports/streaming_sample/20251128/doi_rate_by_year.png)
- ![Top categories, 2025 11 28 (sample streaming)](../reports/streaming_sample/20251128/top_categories.png)

#### 3.2.4 Date 2025 12 05

Directory:

```text
reports/streaming_sample/20251205/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 12 05 (sample streaming)](../reports/streaming_sample/20251205/papers_per_year.png)
- ![DOI rate by year, 2025 12 05 (sample streaming)](../reports/streaming_sample/20251205/doi_rate_by_year.png)
- ![Top categories, 2025 12 05 (sample streaming)](../reports/streaming_sample/20251205/top_categories.png)

#### 3.2.5 Date 2025 12 12

Directory:

```text
reports/streaming_sample/20251212/
```

Tables:

- `by_year.csv`
- `doi_rate_by_year.csv`
- `top_categories.csv`

Figures:

- ![Papers per year, 2025 12 12 (sample streaming)](../reports/streaming_sample/20251212/papers_per_year.png)
- ![DOI rate by year, 2025 12 12 (sample streaming)](../reports/streaming_sample/20251212/doi_rate_by_year.png)
- ![Top categories, 2025 12 12 (sample streaming)](../reports/streaming_sample/20251212/top_categories.png)

---

## 4. Summary

Across standard, complex, and streaming reports, the `reports/` tree provides:

- Tabular CSV summaries for reproducible analysis
- Matching PNG plots for quick visual inspection
- Both full and sample views for most metrics
- Streaming snapshots that mirror standard queries at multiple dates

This file chronicles all known report files and image outputs so they can be easily browsed from the documentation or the web application.
