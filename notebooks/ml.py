#!/usr/bin/env python3
import argparse, os
from typing import List, Tuple
from pyspark.sql import SparkSession, functions as F, Row, Window
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import SparseVector
from src.similarity import _dot_udf
from src.query import query_topk

# ---------- Spark session ----------
def make_spark():
    try:
        from src.utils import get_spark
        return get_spark(app_name="ml_week12_full")
    except Exception:
        return (
            SparkSession.builder
            .appName("ml_week12_full")
            .config("spark.sql.shuffle.partitions", "1024")
            .config("spark.sql.files.maxPartitionBytes", 256 << 20)
            .getOrCreate()
        )

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "query"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split-parquet", help="Path with split=train|test partitions")
    ap.add_argument("--features", help="Root where features were saved by training")
    ap.add_argument("--features-train", help="Path to features for split=train (for --mode query)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", choices=["exact_broadcast", "block_cat"], default="block_cat")
    ap.add_argument("--query-title")
    ap.add_argument("--query-abstract")
    ap.add_argument("--eval-max-test", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ---------- IO helpers ----------
def load_features(spark, features_root, split):
    path = os.path.join(features_root, f"split={split}")
    return spark.read.parquet(path)


def explode_cats(df, col="categories"):
    dtypes = dict(df.dtypes)
    coltype = dtypes.get(col, "")
    if coltype.startswith("array"):
        tmp = df.withColumn("cat", F.explode_outer(F.col(col)))
    else:
        tmp = (
            df.withColumn("_cats", F.split(F.coalesce(F.col(col).cast("string"), F.lit("")), r"\\s+"))
              .withColumn("cat", F.explode_outer(F.col("_cats")))
              .drop("_cats")
        )
    return tmp.where((F.col("cat").isNotNull()) & (F.col("cat") != ""))


# ---------- Exact Top‑K with broadcast (small train only) ----------
def _dot(a: SparseVector, b: SparseVector) -> float:
    if a is None or b is None:
        return 0.0
    ai = dict(zip(a.indices, a.values))
    s = 0.0
    for j, v in zip(b.indices, b.values):
        if j in ai:
            s += ai[j] * v
    return float(s)


def topk_broadcast_exact(spark, test_df, train_df, k: int = 5, exclude_self: bool = True):
    train_local: List[Tuple[str, List[str], SparseVector]] = []
    for r in train_df.select("id_base", "categories", "features").toLocalIterator():
        train_local.append((r["id_base"], r["categories"], r["features"]))
    bc_train = spark.sparkContext.broadcast(train_local)

    def score_partition(iter_rows):
        import heapq as _heapq
        out_rows: List[Row] = []
        train_list = bc_train.value
        for r in iter_rows:
            tid = r["id_base"]; tfeat = r["features"]
            heap: List[Tuple[float, str, List[str]]] = []
            for nid, ncats, nfeat in train_list:
                if exclude_self and nid == tid:
                    continue
                # cosine on L2‑normed vectors == dot
                s = _dot(tfeat, nfeat)
                if len(heap) < k:
                    _heapq.heappush(heap, (s, nid, ncats))
                else:
                    if s > heap[0][0]:
                        _heapq.heapreplace(heap, (s, nid, ncats))
            for rank_idx, (score, nid, ncats) in enumerate(sorted(heap, key=lambda x: (-x[0], x[1])), start=1):
                out_rows.append(Row(test_id=tid, rank=rank_idx, neighbor_id=nid, score=score, neighbor_categories=ncats))
        return iter(out_rows)

    rdd = test_df.select("id_base", "features", "categories").rdd.mapPartitions(score_partition)
    return spark.createDataFrame(rdd)


# ---------- Block-by-category Top‑K (scales) ----------

def topk_block_by_category(test_df, train_df, k: int = 5, exclude_self: bool = True):
    t = explode_cats(test_df.select("id_base", "categories", "features")).select(
        F.col("id_base").alias("tid"), "cat", F.col("features").alias("tfeat")
    )
    c = explode_cats(train_df.select("id_base", "categories", "features")).select(
        F.col("id_base").alias("nid"), "cat", F.col("features").alias("nfeat"), F.col("cat").alias("ncat")
    )
    joined = (
        t.join(F.broadcast(c), on="cat", how="inner")
         .withColumn("score", _dot_udf(F.col("tfeat"), F.col("nfeat")))
    )
    if exclude_self:
        joined = joined.where(F.col("tid") != F.col("nid"))

    w = Window.partitionBy("tid").orderBy(F.desc("score"), F.col("nid"))
    top = (
        joined.withColumn("rank", F.row_number().over(w))
              .where(F.col("rank") <= k)
              .groupBy("tid", "rank", "nid")
              .agg(F.max("score").alias("score"))
              .withColumnRenamed("tid", "test_id")
              .withColumnRenamed("nid", "neighbor_id")
    )
    # Attach neighbor categories for reporting
    neigh = train_df.select(F.col("id_base").alias("nid"), F.col("categories").alias("neighbor_categories"))
    return top.join(neigh, top["neighbor_id"] == neigh["nid"], "left").drop("nid")


# ---------- Attach neighbor arXiv IDs ----------

def _attach_neighbor_paper_id(recs, train):
    id_map = train.select(
        F.col("id_base").alias("nid"),
        F.col("paper_id").alias("neighbor_paper_id")
    )
    return recs.join(id_map, recs.neighbor_id == id_map.nid, "left").drop("nid")


# ---------- Modes ----------

def eval_mode(spark, args):
    os.makedirs(args.out, exist_ok=True)
    _ = PipelineModel.load(args.model_dir)  # parity check

    test_full = load_features(spark, args.features, "test")
    train = load_features(spark, args.features, "train").select("id_base", "categories", "features", "paper_id")

    test = (
        test_full.orderBy(F.rand(args.seed)).limit(args.eval_max_test)
        .select("id_base", "categories", "features")
    ).cache(); _ = test.count()

    if args.strategy == "exact_broadcast":
        recs = topk_broadcast_exact(spark, test, train, k=args.k, exclude_self=True)
    else:
        recs = topk_block_by_category(test, train, k=args.k, exclude_self=True)
    recs = recs.cache(); _ = recs.count()

    recs = _attach_neighbor_paper_id(recs, train).cache(); _ = recs.count()

    recs_to_save = (
        recs.withColumn(
                "neighbor_categories",
                F.array_join(F.col("neighbor_categories").cast("array<string>"), " ")
            )
            .select("test_id", "rank", "neighbor_id", "neighbor_paper_id", "score", "neighbor_categories")
            .orderBy("test_id", "rank")
    )
    recs_out = os.path.join(args.out, "recs_topk.csv")
    recs_to_save.write.mode("overwrite").option("header", True).csv(recs_out)

    # ---- Metrics (same as sample, computed on block results) ----
    test_c = (
        explode_cats(test.select("id_base", "categories").withColumnRenamed("id_base", "tid"))
        .select("tid", F.col("cat").alias("cat_test"))
    )
    train_c = (
        explode_cats(train.select("id_base", "categories").withColumnRenamed("id_base", "nid"))
        .select("nid", F.col("cat").alias("cat_train"))
    )

    overlap = (
        recs.join(train_c, recs.neighbor_id == train_c.nid, "left")
            .join(test_c, recs.test_id == test_c.tid, "left")
            .where(F.col("cat_train").isNotNull() & F.col("cat_test").isNotNull() & (F.col("cat_train") == F.col("cat_test")))
            .groupBy("test_id", "neighbor_id", "rank")
            .agg(F.countDistinct("cat_train").alias("overlap_ct"))
    )

    rel = (
        recs.join(overlap, on=["test_id", "neighbor_id", "rank"], how="left")
            .withColumn("rel", F.when(F.col("overlap_ct") > 0, 1).otherwise(0))
            .cache()
    ); _ = rel.count()

    relevant = (
        test_c.join(train_c, test_c.cat_test == train_c.cat_train, "inner")
               .where(F.col("tid") != F.col("nid"))
               .groupBy("tid").agg(F.countDistinct("nid").alias("relevant_total"))
    )

    rel = (
        rel.join(relevant, rel.test_id == relevant.tid, "left").drop("tid").fillna({"relevant_total": 0})
    ).cache(); _ = rel.count()

    w = Window.partitionBy("test_id").orderBy("rank")
    rel = rel.withColumn("cum_rel", F.sum("rel").over(w)).cache(); _ = rel.count()

    per_test = rel.groupBy("test_id").agg(
        F.sum("rel").alias("sum_rel"),
        F.max("relevant_total").alias("relevant_total"),
    )

    prec = per_test.select("test_id", (F.col("sum_rel") / F.lit(args.k)).alias("precision_at_k"))
    recall = per_test.select(
        "test_id",
        F.when(F.col("relevant_total") > 0, F.col("sum_rel") / F.col("relevant_total")).otherwise(F.lit(None)).alias("recall_at_k"),
    )

    ap = (
        rel.withColumn("prec_at_i", F.col("cum_rel") / F.col("rank"))
           .where(F.col("rel") == 1)
           .groupBy("test_id")
           .agg(F.avg("prec_at_i").alias("ap_at_k"))
    )

    first_hit = rel.where(F.col("rel") == 1).groupBy("test_id").agg(F.min("rank").alias("first_rank"))
    mrr_val = first_hit.select(F.avg(1.0 / F.col("first_rank")).alias("mrr_at_k")).first()["mrr_at_k"]

    cov = rel.groupBy("test_id").agg((F.count("*") >= args.k).cast("int").alias("has_k"))
    coverage_at_k = cov.agg(F.avg("has_k").alias("coverage_at_k")).first()["coverage_at_k"]

    metrics = (
        prec.join(recall, "test_id", "outer").join(ap, "test_id", "outer")
    )

    ild_mean = F.lit(None).alias("intra_list_diversity")  # optional to compute item‑item

    macro = (
        metrics.agg(
            F.avg("precision_at_k").alias("precision_at_k"),
            F.avg("recall_at_k").alias("recall_at_k"),
            F.avg("ap_at_k").alias("map_at_k"),
        )
        .withColumn("mrr_at_k", F.lit(mrr_val))
        .withColumn("coverage_at_k", F.lit(coverage_at_k))
        .withColumn("intra_list_diversity", ild_mean)
    )

    out_csv = os.path.join(args.out, "metrics_at_k.csv")
    macro.write.mode("overwrite").option("header", True).csv(out_csv)

    # Qualitative examples
    test_full = load_features(spark, args.features, "test")
    qual = (
        recs.join(
            test_full.select(F.col("id_base").alias("test_id"), "title", "abstract", "categories"),
            "test_id",
        ).orderBy("test_id", "rank").limit(100)
    )

    qpath = os.path.join(args.out, "qualitative_examples.md")
    rows = [
        (
            r["test_id"],
            r["rank"],
            r["neighbor_id"],
            r["score"],
            r["neighbor_categories"],
        )
        for r in qual.select("test_id", "rank", "neighbor_id", "score", "neighbor_categories").collect()
    ]
    with open(qpath, "w") as f:
        f.write("# Qualitative Examples (Top‑K)\n\n")
        current = None
        for tid, rank, nid, score, ncats in rows:
            if current != tid:
                current = tid
                f.write(f"\n## Test: {tid}\n\n")
            cats_str = " ".join(ncats) if isinstance(ncats, list) else str(ncats)
            f.write(f"- k={int(rank)} → **{nid}** (cos={float(score):.3f}) cats={cats_str}\n")


def query_mode(spark, args):
    os.makedirs(args.out, exist_ok=True)
    model = PipelineModel.load(args.model_dir)
    train = spark.read.parquet(args.features_train)

    recs = query_topk(
        spark,
        model,
        train.select("id_base", "categories", "features"),
        args.query_title or "",
        args.query_abstract or "",
        k=args.k,
    )

    recs = _attach_neighbor_paper_id(recs, train)

    recs_to_save = (
        recs.withColumn(
                "neighbor_categories",
                F.array_join(F.col("neighbor_categories").cast("array<string>"), " ")
            )
            .select("test_id", "rank", "neighbor_id", "neighbor_paper_id", "score", "neighbor_categories")
            .orderBy("rank")
    )

    out_csv = os.path.join(args.out, "query_topK.csv")
    recs_to_save.write.mode("overwrite").option("header", True).csv(out_csv)


# ---------- main ----------

def main():
    args = parse_args()
    spark = make_spark()
    if args.mode == "eval":
        if not args.features:
            raise ValueError("--features is required for eval")
        eval_mode(spark, args)
    else:
        query_mode(spark, args)
    spark.stop()


if __name__ == "__main__":
    main()