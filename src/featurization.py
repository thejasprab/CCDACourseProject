from typing import List, Iterable
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover,
                                CountVectorizer, IDF, NGram)
from pyspark.ml.feature import Normalizer

DEFAULT_DOMAIN_STOPS = [
    "arxiv", "paper", "result", "results", "method", "methods",
    "propose", "proposed", "show", "shows", "using", "based"
]

def _lower_and_strip(df: DataFrame) -> DataFrame:
    return (df
            .withColumn("title_clean", F.lower(F.col("title")))
            .withColumn("abstract_clean", F.lower(F.col("abstract")))
            .withColumn("text", F.concat_ws(" ", "title_clean", "abstract_clean")))

def compute_extra_stopwords(spark: SparkSession, train_df: DataFrame, top_df: int = 200, seed: int = 42) -> List[str]:
    """
    Return the top 'top_df' tokens by document frequency on TRAIN.
    Deterministic: counts distinct id_base per token (no monotonic ID).
    """
    # Ensure id_base exists (split step creates it)
    if "id_base" not in train_df.columns:
        # fall back to 'id' if present
        if "id" in train_df.columns:
            train_df = train_df.withColumn("id_base", F.regexp_replace(F.col("id"), r"v\d+$", ""))
        else:
            raise ValueError("compute_extra_stopwords requires 'id_base' or 'id' in train_df")

    tok = RegexTokenizer(
        inputCol="abstract", outputCol="tmp_tokens",
        pattern=r"[^\\p{L}]+", gaps=True, toLowercase=True
    )
    toks = tok.transform(train_df.select("id_base", "abstract"))

    exploded = (toks
        .select("id_base", F.explode_outer("tmp_tokens").alias("tok"))
        .where(F.length("tok") > 1))

    # One (doc, token) occurrence max
    doc_tok = exploded.dropDuplicates(["id_base", "tok"])

    dfreq = doc_tok.groupBy("tok").agg(F.countDistinct("id_base").alias("df"))
    top = dfreq.orderBy(F.desc("df")).limit(top_df)

    return [r["tok"] for r in top.collect()]


def build_text_pipeline(
    vocab_size: int = 80000,
    min_df: int = 3,
    use_bigrams: bool = False,
    extra_stopwords: Iterable[str] = ()
) -> Pipeline:
    """
    Pipeline:
      RegexTokenizer -> StopWordsRemover(default + extra + domain) -> (optional NGram(2) + concat)
      -> CountVectorizer(minDF, vocabSize) -> IDF -> Normalizer(p=2) as 'features_norm'
    """
    tokenizer = RegexTokenizer(
        inputCol="text", outputCol="tokens",
        pattern=r"[^\\p{L}]+", gaps=True, toLowercase=True)

    remover = StopWordsRemover(
        inputCol="tokens", outputCol="tokens_sw_removed",
        stopWords=sorted(set(StopWordsRemover.loadDefaultStopWords("english") +
                             list(DEFAULT_DOMAIN_STOPS) + list(extra_stopwords)))
    )

    stages = [
        # title + abstract => text
        # (use small SQL stage before to avoid a Python UDF)
    ]

    # We'll assemble 'text' upstream before calling the pipeline's .fit()
    # but keep tokenizer as the first stage for clarity.
    stages.append(tokenizer)
    stages.append(remover)

    final_tokens_col = "tokens_sw_removed"

    if use_bigrams:
        bigr = NGram(n=2, inputCol=final_tokens_col, outputCol="tokens_bi")
        stages.append(bigr)
        # concat unigrams + bigrams (keep counts — use array_concat)
        stages.append(_ConcatArrays(inputCol1=final_tokens_col, inputCol2="tokens_bi", outputCol="tokens_all"))
        final_tokens_col = "tokens_all"

    vectorizer = CountVectorizer(
        inputCol=final_tokens_col, outputCol="tf", vocabSize=vocab_size, minDF=min_df
    )
    idf = IDF(inputCol="tf", outputCol="features_tfidf")
    norm = Normalizer(inputCol="features_tfidf", outputCol="features_norm", p=2.0)

    return Pipeline(stages=[tokenizer, remover] + (stages[2:] if len(stages) > 2 else []) + [vectorizer, idf, norm])

# Helper transformer: concat two array<string> columns preserving multiplicity.
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params
from pyspark.sql.types import ArrayType, StringType, StructType

class _ConcatArrays(Transformer):
    def __init__(self, inputCol1: str, inputCol2: str, outputCol: str):
        super().__init__()
        self.inputCol1 = inputCol1
        self.inputCol2 = inputCol2
        self.outputCol = outputCol

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.withColumn(self.outputCol, F.array_concat(F.col(self.inputCol1), F.col(self.inputCol2)))
