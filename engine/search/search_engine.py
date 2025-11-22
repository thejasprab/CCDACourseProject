from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
from scipy.sparse import load_npz
from pyspark.ml.linalg import SparseVector
from pyspark.sql import functions as F

from engine.utils.spark_utils import get_spark
from engine.ml.model_loader import load_model_and_features
from engine.search.vectorize import vectorize_query, query_topk


def _cosine_sparse(a: SparseVector, b: SparseVector) -> float:
    """
    Cosine similarity for L2 normalized SparseVector inputs.

    Since TF-IDF vectors are already L2-normalized in this project,
    cosine similarity = dot product.
    """
    if a is None or b is None:
        return 0.0
    ai = dict(zip(a.indices, a.values))
    s = 0.0
    for j, v in zip(b.indices, b.values):
        if j in ai:
            s += ai[j] * v
    return float(s)


def _normalize_categories_for_output(cats) -> list | None:
    """
    Make sure categories is either:
      - None
      - a plain Python list of strings
    """
    if cats is None:
        return None

    if isinstance(cats, float) and np.isnan(cats):
        return None

    if isinstance(cats, list):
        return cats

    try:
        return list(cats)
    except TypeError:
        return [str(cats)]


class SearchEngine:
    """
    Search wrapper for either 'sample' or 'full' mode.

    - Spark is used offline to compute features and train the TF-IDF model.
    - At runtime:
        * sample mode:
            - load all features into Python once
            - do cosine similarity in pure Python / NumPy (fast)
        * full mode:
            - load a SciPy CSR index built offline
            - do sparse matrix * dense vector at query time (fast)
    """

    def __init__(self, mode: str = "sample", spark=None):
        if mode not in {"sample", "full"}:
            raise ValueError("mode must be 'sample' or 'full'")

        self.mode = mode
        self.spark = spark or get_spark(f"search_{mode}")

        # Load trained TF-IDF pipeline and precomputed features (Spark DataFrame)
        self.model, self.features = load_model_and_features(self.spark, mode)

        if self.mode == "sample":
            self._init_sample_index()
        else:
            self._init_full_index()

    # -------------------------------------------------------------------------
    # Sample mode local index (what you already had working)
    # -------------------------------------------------------------------------

    def _init_sample_index(self) -> None:
        """
        Build an in memory index from the sample features parquet.

        We keep:
          - metadata arrays
          - list of SparseVector feature objects
        """
        print("[search] initializing local index for SAMPLE mode")

        pdf = (
            self.features.select(
                "id_base",
                "paper_id",
                "title",
                "abstract",
                "categories",
                "year",
                "features",
            )
            .toPandas()
        )

        self._sample_ids = pdf["id_base"].astype(str).to_numpy()
        self._sample_paper_ids = pdf["paper_id"].astype(str).to_numpy()
        self._sample_titles = pdf["title"].astype(str).to_numpy()
        self._sample_abstracts = pdf["abstract"].astype(str).to_numpy()
        self._sample_categories = pdf["categories"].to_numpy()
        self._sample_years = pdf["year"].to_numpy()
        self._sample_vecs: List[SparseVector] = pdf["features"].tolist()

        self._has_sample_index = True
        print(f"[search] sample index ready - {len(self._sample_vecs)} documents")

    # -------------------------------------------------------------------------
    # Full mode CSR index
    # -------------------------------------------------------------------------

    def _init_full_index(self) -> None:
        """
        Load the CSR index and metadata for the FULL dataset.

        This assumes `pipelines.build_full_index` has been run and produced
        the artifacts under data/processed/full_index/.
        """
        print("[search] initializing CSR index for FULL mode")

        base = "data/processed/full_index"
        csr_path = f"{base}/full_index_csr.npz"
        ids_path = f"{base}/full_index_ids.npy"
        pid_path = f"{base}/full_index_paper_ids.npy"
        titles_path = f"{base}/full_index_titles.npy"
        abs_path = f"{base}/full_index_abstracts.npy"
        cats_path = f"{base}/full_index_categories.npy"
        years_path = f"{base}/full_index_years.npy"

        self._full_mat = load_npz(csr_path)
        self._full_ids = np.load(ids_path, allow_pickle=True)
        self._full_paper_ids = np.load(pid_path, allow_pickle=True)
        self._full_titles = np.load(titles_path, allow_pickle=True)
        self._full_abstracts = np.load(abs_path, allow_pickle=True)
        self._full_categories = np.load(cats_path, allow_pickle=True)
        self._full_years = np.load(years_path, allow_pickle=True)

        if self._full_mat.shape[0] != self._full_ids.shape[0]:
            raise RuntimeError("CSR matrix row count does not match ids array length")

        self._has_full_index = True
        print(
            f"[search] full CSR index ready - "
            f"{self._full_mat.shape[0]} docs, dim={self._full_mat.shape[1]}"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def search(
        self, title: str = "", abstract: str = "", k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Main search entry point used by the Flask app.

        Sample mode: use in memory index.
        Full mode  : use CSR index.
        """
        if self.mode == "sample" and getattr(self, "_has_sample_index", False):
            return self._search_sample_local(title=title, abstract=abstract, k=k)
        if self.mode == "full" and getattr(self, "_has_full_index", False):
            return self._search_full_csr(title=title, abstract=abstract, k=k)

        # Fallback: Spark based brute force if index is missing
        return self._search_spark_fallback(title=title, abstract=abstract, k=k)

    # -------------------------------------------------------------------------
    # Sample mode - fast local inference
    # -------------------------------------------------------------------------

    def _search_sample_local(
        self, title: str = "", abstract: str = "", k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fast inference for SAMPLE mode.

        1. Vectorize query with Spark TF-IDF pipeline.
        2. Cosine similarity against local SparseVector list.
        """
        qvec = vectorize_query(self.spark, self.model, title, abstract)

        scores = np.empty(len(self._sample_vecs), dtype=np.float32)
        for i, v in enumerate(self._sample_vecs):
            scores[i] = _cosine_sparse(qvec, v)

        k = max(1, min(k, len(scores)))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_idx, start=1):
            cats_raw = self._sample_categories[idx]
            cats = _normalize_categories_for_output(cats_raw)

            year_raw = self._sample_years[idx]
            year_val = (
                int(year_raw)
                if year_raw is not None
                and not (isinstance(year_raw, float) and np.isnan(year_raw))
                else None
            )

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "neighbor_id": str(self._sample_ids[idx]),
                    "paper_id": str(self._sample_paper_ids[idx]),
                    "title": str(self._sample_titles[idx]),
                    "abstract": str(self._sample_abstracts[idx]),
                    "categories": cats,
                    "year": year_val,
                }
            )

        return results

    # -------------------------------------------------------------------------
    # Full mode - CSR based inference
    # -------------------------------------------------------------------------

    def _search_full_csr(
        self, title: str = "", abstract: str = "", k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fast inference for FULL mode using CSR index.

        1. Vectorize query with Spark TF-IDF pipeline to a SparseVector.
        2. Convert query to dense float32 vector (dimension = vocab size).
        3. Compute scores = CSR_matrix @ query_dense.
        4. Take top-k, return metadata from preloaded arrays.
        """
        qvec = vectorize_query(self.spark, self.model, title, abstract)

        dim = self._full_mat.shape[1]
        q_dense = np.zeros(dim, dtype=np.float32)
        # qvec.indices and qvec.values define the sparse vector
        q_dense[qvec.indices] = qvec.values

        scores = self._full_mat.dot(q_dense).astype(np.float32)

        n = scores.shape[0]
        k = max(1, min(k, n))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_idx, start=1):
            cats_raw = self._full_categories[idx]
            cats = _normalize_categories_for_output(cats_raw)

            year_raw = self._full_years[idx]
            year_val = (
                int(year_raw)
                if year_raw is not None
                and not (isinstance(year_raw, float) and np.isnan(year_raw))
                else None
            )

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "neighbor_id": str(self._full_ids[idx]),
                    "paper_id": str(self._full_paper_ids[idx]),
                    "title": str(self._full_titles[idx]),
                    "abstract": str(self._full_abstracts[idx]),
                    "categories": cats,
                    "year": year_val,
                }
            )

        return results

    # -------------------------------------------------------------------------
    # Fallback Spark search (in case index is missing)
    # -------------------------------------------------------------------------

    def _search_spark_fallback(
        self, title: str = "", abstract: str = "", k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Spark based search - used as a fallback if local index is missing.

        This still does brute-force cosine@k over the corpus using Spark,
        so it can be slow on a single machine but preserves old behavior.
        """
        base = self.features.select(
            "id_base",
            "paper_id",
            "title",
            "abstract",
            "categories",
            "year",
            "features",
        )
        feats = base.select("id_base", "categories", "features")

        recs = query_topk(
            self.spark,
            self.model,
            feats,
            query_title=title,
            query_abstract=abstract,
            k=k,
        )

        meta = base.select(
            F.col("id_base").alias("neighbor_id_meta"),
            "paper_id",
            "title",
            "abstract",
            "categories",
            "year",
        )

        joined = (
            recs.join(
                meta,
                recs.neighbor_id == meta.neighbor_id_meta,
                "left",
            )
            .drop("neighbor_id_meta")
            .orderBy("rank")
        )

        out: List[Dict[str, Any]] = []
        for r in joined.collect():
            cats = _normalize_categories_for_output(r["categories"])
            out.append(
                {
                    "rank": int(r["rank"]),
                    "score": float(r["score"]),
                    "neighbor_id": r["neighbor_id"],
                    "paper_id": r["paper_id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "categories": cats,
                    "year": r["year"],
                }
            )
        return out
