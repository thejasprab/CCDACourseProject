#!/usr/bin/env python3
import argparse, os, itertools
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import SparseVector
from src.similarity import topk_exact, _dot_udf
from src.query import query_topk

def make_spark():
    try:
        from src.utils import get_spark
        return get_spark(app_name="ml_sample_week12")
    except Exception:
        return (SparkSession.builder
                .appName("ml_sample_week12")
                .config("spark.sql.shuffle.partitions", "64")
                .getOrCreate())

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "query"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split-parquet", help="Path with split=train|test partitions")
    ap.add_argument("--features", help="Root where features were saved by training")
    ap.add_argument("--features-train", help="Path to features for split=train (for --mode query)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--strategy", choices=["exact"], default="exact")
    ap.add_argument("--query-title")
    ap.add_argument("--query-abstract")
    return ap.parse_args()

def load_features(spark, features_root, split):
    path = os.path.join(features_root, f"split={split}")
    return spark.read.parquet(path)

def explode_cats(df, col="categories"):
    return df.withColumn("cat", F.explode_outer(F.split(F.coalesce(F.col(col), F.lit("")), r"\s+"))).where(F.col("cat") != "")

def eval_mode(spark, args):
    os.makedirs(args.out, exist_ok=True)
    model = PipelineModel.load(args.model_dir)
    test = load_features(spark, args.features, "test")
    train = load_features(spark, args.features, "train")

    # Top-K exact cosine
    recs = topk_exact(
        test.select("id_base", "categories", "features"),
        train.select("id_base", "categories", "features"),
        k=args.k, exclude_self=True
    )
    
    # Save recs (stringify arrays for CSV)
    recs_to_save = (recs
        .withColumn("neighbor_categories",
                    F.array_join(F.col("neighbor_categories").cast("array<string>"), " ")))

    # Save recs
    recs_out = os.path.join(args.out, "recs_topk.csv")
    (recs_to_save
    .orderBy("test_id", "rank")
    .coalesce(1)
    .write.mode("overwrite").option("header", True).csv(recs_out))

    # Build relevance: any category overlap
    test_c = explode_cats(test.select("id_base", "categories").withColumnRenamed("id_base","tid"))
    train_c = explode_cats(train.select("id_base", "categories").withColumnRenamed("id_base","nid"))
    overlap = (recs.join(train_c, recs.neighbor_id == train_c.nid, "left")
                    .join(test_c, recs.test_id == test_c.tid, "left")
                    .where(F.col("cat").isNotNull())
                    .groupBy("test_id","neighbor_id","rank")
                    .agg(F.countDistinct("cat").alias("overlap_ct")))
    rel = recs.join(overlap, on=["test_id","neighbor_id","rank"], how="left")\
              .withColumn("rel", F.when(F.col("overlap_ct") > 0, F.lit(1)).otherwise(F.lit(0)))

    # total relevant per test (in TRAIN): join test cats to train cats (count unique neighbors)
    relevant = (test_c.join(train_c, "cat", "inner")
                      .where(F.col("tid") != F.col("nid"))
                      .groupBy("tid").agg(F.countDistinct("nid").alias("relevant_total")))

    rel = rel.join(relevant, rel.test_id == relevant.tid, "left")\
             .drop("tid").fillna({"relevant_total": 0})

    # Metrics@K per test
    w = Window.partitionBy("test_id").orderBy("rank")
    rel = rel.withColumn("cum_rel", F.sum("rel").over(w))

    # Precision@K: sum rel / K
    prec = rel.groupBy("test_id").agg((F.sum("rel")/F.lit(args.k)).alias("precision_at_k"))

    # Recall@K: sum rel / relevant_total (clip denominator)
    recall = rel.groupBy("test_id").agg(
        F.when(F.col("relevant_total") > 0,
               F.sum("rel")/F.col("relevant_total")).otherwise(F.lit(None)).alias("recall_at_k"))

    # MAP@K
    ap = (rel.withColumn("prec_at_i", F.col("cum_rel")/F.col("rank"))
              .where(F.col("rel") == 1)
              .groupBy("test_id")
              .agg(F.avg("prec_at_i").alias("ap_at_k")))
    # If no relevant, ap_at_k null; keep as null
    # MRR
    first_hit = rel.where(F.col("rel") == 1).groupBy("test_id").agg(F.min("rank").alias("first_rank"))
    mrr = first_hit.select(F.avg(1.0/F.col("first_rank")).alias("mrr_at_k"))

    # Coverage@K: had >=K recs
    cov = rel.groupBy("test_id").agg((F.count("*") >= args.k).cast("int").alias("has_k"))
    coverage_at_k = cov.agg(F.avg("has_k").alias("coverage_at_k")).first()["coverage_at_k"]

    # Intra-list diversity: mean pairwise cosine among top-K (lower better).
    # We need neighbor-neighbor cosine; fetch features for neighbors and self-join
    neigh_feats = train.select(F.col("id_base").alias("nid"), F.col("features").alias("nfeat"))
    rk = (recs.join(neigh_feats, recs.neighbor_id == neigh_feats.nid, "left")
               .select("test_id","rank","neighbor_id", F.col("nfeat").alias("feat")))
    a = rk.alias("a"); b = rk.alias("b")
    pairs = (a.join(b, (a.test_id==b.test_id) & (a.rank < b.rank))
               .withColumn("pair_cosine", _dot_udf(F.col("a.feat"), F.col("b.feat"))))
    ild_mean = pairs.groupBy("test_id").agg(F.avg("pair_cosine").alias("ild_mean"))

    # Collect macro metrics
    metrics = (prec.join(recall, "test_id", "outer")
                    .join(ap, "test_id", "outer")
                    .join(ild_mean, "test_id", "left"))

    # Macro averages
    macro = metrics.agg(
        F.avg("precision_at_k").alias("precision_at_k"),
        F.avg("recall_at_k").alias("recall_at_k"),
        F.avg("ap_at_k").alias("map_at_k"),
        F.avg("ild_mean").alias("intra_list_diversity")
    ).withColumn("mrr_at_k", F.lit(mrr.first()["mrr_at_k"]))\
     .withColumn("coverage_at_k", F.lit(coverage_at_k))

    out_csv = os.path.join(args.out, "metrics_at_k.csv")
    (macro.coalesce(1)
          .write.mode("overwrite").option("header", True).csv(out_csv))

    # Qualitative examples (10)
    qual = (recs.join(test.select(F.col("id_base").alias("test_id"),
                                  "title","abstract","categories"), "test_id")
                 .orderBy("test_id","rank"))
    qual_pd = qual.limit(30).toPandas()  # small
    qpath = os.path.join(args.out, "qualitative_examples.md")
    with open(qpath, "w") as f:
        f.write("# Qualitative Examples (Top-K)\n\n")
        for tid in qual_pd["test_id"].unique()[:10]:
            subset = qual_pd[qual_pd["test_id"]==tid]
            f.write(f"## Test: {tid}\n\n")
            row = test.where(F.col("id_base")==tid).select("title","categories").first()
            f.write(f"**Title:** {row['title']}\n\n")
            f.write(f"**Categories:** {row['categories']}\n\n")
            for _,r in subset.iterrows():
                f.write(f"- k={int(r['rank'])} → **{r['neighbor_id']}** (cos={r['score']:.3f}) cats={r['neighbor_categories']}\n")
            f.write("\n")

def query_mode(spark, args):
    os.makedirs(args.out, exist_ok=True)
    model = PipelineModel.load(args.model_dir)
    train = spark.read.parquet(args.features_train)
    recs = query_topk(
        spark, model,
        train.select("id_base","categories","features"),
        args.query_title or "", args.query_abstract or "", k=args.k
    )

    # Stringify arrays for CSV
    recs_to_save = recs.withColumn(
        "neighbor_categories",
        F.array_join(F.col("neighbor_categories").cast("array<string>"), " ")
    )

    out_csv = os.path.join(args.out, "query_topK.csv")
    (recs_to_save.orderBy("rank")
    .coalesce(1)
    .write.mode("overwrite").option("header", True).csv(out_csv))

def main():
    args = parse_args()
    spark = make_spark()
    if args.mode == "eval":
        if not (args.split_parquet and args.features):
            # features are required (we don't refit here)
            pass
        eval_mode(spark, args)
    else:
        query_mode(spark, args)
    spark.stop()

if __name__ == "__main__":
    main()
