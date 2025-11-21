from pyspark.sql import functions as F
from pyspark.sql.column import Column


def clean_text(col: Column) -> Column:
    """Trim and collapse whitespace."""
    return F.regexp_replace(F.trim(col), r"\s+", " ")


def parse_year_from_datestr(col: Column) -> Column:
    """Extract the first 4-digit year as int."""
    return F.regexp_extract(col.cast("string"), r"(\d{4})", 1).cast("int")


def extract_primary_category(categories_col: Column) -> Column:
    """First whitespace-delimited token is the primary category."""
    return F.split(F.coalesce(categories_col, F.lit("")), r"\s+")[0]


def split_categories(categories_col: Column) -> Column:
    """Split on whitespace into array, dropping empties."""
    arr = F.split(F.coalesce(categories_col, F.lit("")), r"\s+")
    return F.filter(arr, lambda x: x != "")


def normalize_authors(authors_col: Column) -> Column:
    """
    Normalize authors string into an array:
      - Replace ' and ' with commas
      - Normalize comma/whitespace
      - Trim and drop empties
    """
    replaced = F.regexp_replace(F.coalesce(authors_col, F.lit("")), r"\s+and\s+", ",")
    replaced = F.regexp_replace(replaced, r"\s*,\s*", ",")
    arr = F.split(replaced, ",")
    arr = F.transform(arr, lambda x: F.trim(x))
    return F.filter(arr, lambda x: x != "")


def lower(col: Column) -> Column:
    """Lowercase after cleanup."""
    return F.lower(clean_text(col))
