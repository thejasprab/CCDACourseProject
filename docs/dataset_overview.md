# Dataset Overview

Sparxiv is built on top of the **arXiv metadata** snapshot published by Cornell University and mirrored on **Kaggle** as a JSON Lines file. The Kaggle snapshot used here contains metadata for **3M+ scholarly papers** across physics, mathematics, computer science, statistics, quantitative biology, economics, and related fields.

This document describes the field level schema used in the project, covering both the **raw Kaggle fields** and the **derived columns** created during ingestion, analytics, and model training.

The goal is to have a single unified reference for all important columns you will see in:

- ingested Parquet datasets under `data/processed/`
- ML feature tables under `data/processed/features_*`
- analytics views such as `papers` and `papers_enriched`

---

## 1. Source Dataset

- **Upstream source**: Kaggle dataset `Cornell-University/arxiv`
- **Approximate size**: 3M+ papers (metadata only, no PDFs)
- **Raw file name**: `arxiv-metadata-oai-snapshot.json`
- **Format**: JSON Lines (one paper per line)
- **Project locations**:
  - Raw full snapshot: `data/raw/arxiv-metadata-oai-snapshot.json`
  - Sample JSONL: `data/sample/arxiv-sample.jsonl` (head slice generated from the full file)

The ingestion pipelines read this JSONL and write cleaned, normalized Parquet datasets for both the full and sample flows.

---

## 2. Unified Schema Table

The table below documents all key fields that appear across:

- raw Kaggle JSON
- ingested Parquet (`data/processed/arxiv_full`, `data/processed/arxiv_sample`)
- analytics views (`papers`, `papers_enriched`)
- ML feature tables (`data/processed/features_full`, `data/processed/features_sample`)

This is the main schema reference for Sparxiv.

| Column name       | Type                     | Origin                                         | Description                                                                                                                                                           | Used in                                                                                 |
| ----------------- | ------------------------ | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `id`              | string                   | Raw Kaggle field                               | Original arXiv identifier for this record, usually including a version suffix, for example `0704.0001v1`.                                                             | Ingested Parquet, training input, linking back to the raw snapshot.                    |
| `arxiv_id`        | string                   | Derived during ingestion if needed             | Alternate name for the arXiv id used by some older code; if present it mirrors `id`.                                                                                 | Training helper in `train_model` to build `id_base` and `paper_id`.                    |
| `id_base`         | string                   | Derived in ingestion and training              | Versionless identifier obtained by stripping trailing `v\d+` from `id` or `arxiv_id`, for example `0704.0001`. Used as a stable key across all versions of a paper.   | Ingested Parquet, feature tables, complex analytics, sample and full search indices.   |
| `paper_id`        | string                   | Derived in training                            | Canonical per record identifier; usually equal to the full arXiv id (including version).                                                                              | Feature tables, full CSR index metadata, search results.                               |
| `submitter`       | string                   | Raw Kaggle field                               | Free form name of the submitter as provided in the metadata.                                                                                                         | Ingested Parquet for inspection and potential future analyses.                         |
| `authors`         | string                   | Raw Kaggle field                               | Raw author list as a single string, often comma separated and sometimes inconsistently formatted.                                                                     | Ingested Parquet, fallback input for building `authors_list`.                          |
| `authors_parsed`  | array of arrays          | Raw Kaggle field                               | Structured author representation `[last, first, middle]` per author, when available.                                                                                 | Preferred source when constructing `authors_list` and derived author statistics.       |
| `authors_list`    | array<string>            | Derived in ingestion                            | Normalized human readable list of author names created from `authors_parsed` where possible, otherwise from `authors`.                                               | Complex analytics (collaboration over time, author lifecycle, category migration).     |
| `num_authors`     | int                      | Derived in ingestion                            | Number of authors per paper, usually `size(authors_list)`.                                                                                                            | Collaboration statistics and author level analyses.                                    |
| `title`           | string                   | Raw Kaggle field                               | Paper title as provided, often containing LaTeX markup or special characters.                                                                                        | Ingested Parquet, TF-IDF text construction, search output.                              |
| `title_len`       | int                      | Derived in ingestion                            | Length of the title in characters, used as a simple proxy for title verbosity.                                                                                       | Standard text length summaries and descriptive statistics.                             |
| `abstract`        | string                   | Raw Kaggle field                               | Paper abstract as provided, usually a few paragraphs of scientific text with possible LaTeX markup.                                                                  | Ingested Parquet, TF-IDF text construction, complex analytics on readability.          |
| `abstract_len`    | int                      | Derived in ingestion                            | Length of the abstract in characters, used for histograms, deciles, and length based correlation analyses.                                                           | Standard queries, complex abstract length vs versions correlations.                    |
| `comments`        | string or null           | Raw Kaggle field                               | Free form comments field, often containing page counts, figure counts, or venue notes.                                                                               | Ingested Parquet for inspection; not used directly in current analytics.               |
| `journal_ref`     | string or null           | Derived from raw `journal-ref`                 | Journal or conference reference, normalized to a snake case column name.                                                                                             | Ingested Parquet; potential hook for future link out or quality analyses.              |
| `report_no`       | string or null           | Derived from raw `report-no`                   | Technical report identifier when provided, normalized to snake case.                                                                                                 | Ingested Parquet; potential hook for external linking.                                 |
| `license`         | string or null           | Raw Kaggle field                               | License string when present; often `null` or omitted in many records.                                                                                                | Ingested Parquet; not used in current analytics but important for understanding reuse. |
| `doi`             | string or null           | Raw Kaggle field                               | Digital Object Identifier string, often empty for older records.                                                                                                     | Ingested Parquet, DOI completeness analyses, DOI vs versions correlation.              |
| `has_doi`         | boolean                  | Derived in ingestion                            | Flag indicating whether `doi` is non null and non empty after trimming.                                                                                              | Standard completeness queries, complex DOI related analyses.                           |
| `versions`        | array of structs         | Raw Kaggle field                               | List of version entries for this arXiv record, including version labels and timestamps where available.                                                              | Ingested Parquet, used to derive `n_versions` and sometimes `submitted_date`.          |
| `n_versions`      | int                      | Derived in `papers_enriched`                   | Number of versions for this paper, usually `size(versions)` or `1` when version metadata is missing.                                                                 | Complex analytics on version behavior, DOI vs versions, category stability.            |
| `categories`      | string or array<string>  | Raw Kaggle field then normalized               | Original arXiv category string, space separated in the raw data. Normalized to an array of strings in the feature tables and some ingested schemas.                  | Ingested Parquet, feature tables, search output, category based analytics.             |
| `categories_list` | array<string>            | Derived in `papers_enriched`                   | Guaranteed array representation of categories, built from `categories` or `primary_category` depending on availability.                                             | Category co occurrence graphs and any multi category analysis.                         |
| `primary_category`| string                   | Derived in ingestion                            | First category token from the `categories` string, treated as the paper's main subject label.                                                                        | Standard category statistics, rising or declining topics, category stability analyses. |
| `update_date`     | date                     | Raw Kaggle field                               | Date of latest update in `YYYY-MM-DD` format, parsed to Spark date type.                                                                                            | Used to derive `year` when no better timestamp is available.                           |
| `submitted_date`  | timestamp or date        | Raw or derived from `versions`                 | Original submission timestamp when available, or earliest version date in the versions list.                                                                         | Weekday submission patterns and time based analyses.                                   |
| `year`            | int                      | Derived in ingestion                            | Year extracted from `update_date` or `submitted_date`, used for partitioning and any time series based analytics.                                                    | Dataset partitioning, standard queries, complex temporal analytics, model features.    |
| `source_date`     | string                   | Derived in streaming                           | Date stamp (YYYYMMDD) derived from the filename of streaming input drops, for example `arxiv-sample-20251121.jsonl`.                                                | Streaming reports to group distinct microbatches.                                      |
| `doi_int`         | int (0 or 1)             | Derived in `papers_enriched`                   | Integer indicator mirroring `has_doi` for numeric correlation and grouping, where `1` means DOI present.                                                             | DOI vs versions correlation and related complex analyses.                              |
| `text`            | string                   | Derived in training                            | Lowercased concatenation of `title` and `abstract`, used as the input to the TF-IDF text pipeline.                                                                   | Training inputs for TF-IDF model and query vectorization.                              |
| `features`        | VectorUDT (sparse)       | Derived in training                            | L2 normalized TF-IDF feature vector produced by the ML pipeline, aliased from `features_norm`.                                                                      | Feature Parquet, full CSR index builder, search engine similarity computations.        |
| `features_norm`   | VectorUDT (sparse)       | Derived in training pipeline internals         | Internal pipeline output for normalized TF-IDF features before renaming; persisted as `features` in the feature Parquet tables.                                      | Only used inside the ML pipeline; exposed as `features` in saved outputs.             |

---

## 3. Original Kaggle JSON Fields

The original JSON objects in `arxiv-metadata-oai-snapshot.json` have a core set of raw fields under the root. A typical record looks like:

```json
{
  "id": "0704.0001",
  "submitter": "Pavel Nadolsky",
  "authors": "C. Bal\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan",
  "title": "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies",
  "comments": "37 pages, 15 figures; published version",
  "journal-ref": "Phys.Rev.D76:013009,2007",
  "doi": "10.1103/PhysRevD.76.013009",
  "report-no": "ANL-HEP-PR-07-12",
  "categories": "hep-ph",
  "license": null,
  "abstract": " A fully differential calculation in perturbative quantum chromodynamics is presented ... ",
  "versions": [
    { "...": "..." },
    { "...": "..." }
  ],
  "update_date": "2008-11-26",
  "authors_parsed": [
    ["Balazs", "C.", ""],
    ["Berger", "E. L.", ""],
    ["Nadolsky", "P. M.", ""],
    ["Yuan", "C.-P.", ""]
  ]
}
```

The table below lists only these **original Kaggle fields** and how Sparxiv interprets them before any derived columns are added.

| Field name      | Type              | Example value                                                                                                               | Description                                                                                          |
| --------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `id`            | string            | `"0704.0001"` or `"0704.0001v1"`                                                                                            | arXiv identifier for the record. Some snapshots include version suffix here, others use separate versions. |
| `submitter`     | string            | `"Pavel Nadolsky"`                                                                                                          | Name of the person who submitted the paper.                                                          |
| `authors`       | string            | `"C. Bal\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan"`                                                                  | Raw, comma separated author list as a single string.                                                 |
| `title`         | string            | `"Calculation of prompt diphoton production cross sections at Tevatron and LHC energies"`                                   | Paper title as free text, often including LaTeX markup.                                             |
| `comments`      | string or null    | `"37 pages, 15 figures; published version"`                                                                                | Additional comments such as length, figures, or publication notes.                                  |
| `journal-ref`   | string or null    | `"Phys.Rev.D76:013009,2007"`                                                                                                | Journal or conference reference if available.                                                       |
| `doi`           | string or null    | `"10.1103/PhysRevD.76.013009"`                                                                                              | Digital Object Identifier. Often empty for older entries.                                           |
| `report-no`     | string or null    | `"ANL-HEP-PR-07-12"`                                                                                                        | Alternate report or preprint number.                                                                |
| `categories`    | string            | `"hep-ph"` or `"cs.LG cs.AI"`                                                                                               | Space separated list of arXiv subject categories.                                                   |
| `license`       | string or null    | `null` or a license string                                                                                                  | License information when available. Often null.                                                     |
| `abstract`      | string            | `" A fully differential calculation in perturbative quantum chromodynamics is presented ... "`                              | Paper abstract, often including LaTeX and math notation.                                            |
| `versions`      | array of objects  | `[{"version": "v1", ...}, {"version": "v2", ...}]`                                                                          | Per version metadata including version labels and dates.                                            |
| `update_date`   | string (date)     | `"2008-11-26"`                                                                                                              | Date of the latest update for this record in `YYYY-MM-DD` format.                                   |
| `authors_parsed`| array of arrays   | `[["Balazs", "C.", ""], ["Berger", "E. L.", ""], ["Nadolsky", "P. M.", ""], ["Yuan", "C.-P.", ""]]`                         | Parsed author representation, where each inner array contains last name, first name, and middle.    |

All other columns in the unified schema table are derived from these original raw fields during the ingestion and transformation steps.
