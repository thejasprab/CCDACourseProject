# src/transformations.py
from pyspark.sql import DataFrame, functions as F
from src.utils import clean_text, parse_year_from_datestr, extract_primary_category, split_categories, normalize_authors, lower

def select_and_standardize(df: DataFrame) -> DataFrame:
    cols = df.columns
    base = df.select(
        F.col("id").alias("arxiv_id"),
        F.col("title"),
        F.col("abstract"),
        F.col("categories"),
        F.col("doi"),
        F.col("journal-ref").alias("journal_ref"),
        F.col("comments"),
        F.col("submitter"),
        F.col("update_date"),
        F.col("versions"),
        *([F.col("authors")] if "authors" in cols else []),
        *([F.col("authors_parsed")] if "authors_parsed" in cols else []),
    )
    base = base.withColumn("title_clean", clean_text(F.col("title")))
    base = base.withColumn("abstract_clean", clean_text(F.col("abstract")))
    base = base.withColumn("title_lower", lower(F.col("title")))
    base = base.withColumn("abstract_lower", lower(F.col("abstract")))
    base = base.withColumn("primary_category", extract_primary_category(F.col("categories")))
    base = base.withColumn("category_list", split_categories(F.col("categories")))
    base = base.withColumn("year", parse_year_from_datestr(F.col("update_date")))

    if "authors_parsed" in cols:
        base = base.withColumn(
            "authors_list",
            F.expr("transform(authors_parsed, x -> array_join(reverse(x), ' '))")
        )
    elif "authors" in cols:
        base = base.withColumn("authors_list", normalize_authors(F.col("authors")))
    else:
        base = base.withColumn("authors_list", F.array())

    base = base.withColumn("abstract_len", F.length("abstract_clean"))
    base = base.withColumn("title_len", F.length("title_clean"))
    base = base.withColumn("n_authors", F.size("authors_list"))
    base = base.withColumn("n_categories", F.size("category_list"))
    return base

def filter_for_quality(df: DataFrame, min_abstract_len: int = 40) -> DataFrame:
    return (
        df
        .where(F.col("arxiv_id").isNotNull() & (F.col("arxiv_id") != ""))
        .where(F.col("title_clean").isNotNull() & (F.length("title_clean") > 0))
        .where(F.col("abstract_clean").isNotNull() & (F.col("abstract_len") >= min_abstract_len))
    )

def add_eda_helpers(df: DataFrame) -> DataFrame:
    return df.withColumn("has_doi", (F.col("doi").isNotNull()) & (F.col("doi") != ""))

def transform_all(df: DataFrame, min_abstract_len: int = 40) -> DataFrame:
    return add_eda_helpers(filter_for_quality(select_and_standardize(df), min_abstract_len))
