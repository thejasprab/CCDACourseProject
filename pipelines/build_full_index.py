# pipelines/build_full_index.py

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

import pyarrow.dataset as ds


def build_full_index(
    features_path: str = "data/processed/features_full",
    out_dir: str = "data/processed/full_index",
    batch_size: int = 512,
    max_docs: int | None = None,
) -> None:
    """
    Build a sparse CSR index for the FULL dataset's TF-IDF features
    WITHOUT using Spark (no JVM, no executor heap).

    Inputs:
      - features_path: Parquet directory written by pipelines.train_full

    Outputs under out_dir:
      - full_index_csr.npz          : CSR matrix (num_docs, vocab_dim)
      - full_index_ids.npy          : id_base per row
      - full_index_paper_ids.npy    : paper_id per row
      - full_index_titles.npy       : title per row
      - full_index_abstracts.npy    : abstract per row
      - full_index_categories.npy   : categories (object array, list per row)
      - full_index_years.npy        : year per row

    Notes:
      - batch_size controls how many rows are processed per Arrow batch.
      - max_docs (if not None) lets you cap the number of docs indexed
        to keep memory & runtime manageable.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[index-full] reading features from {features_path} via pyarrow.dataset")

    dataset = ds.dataset(features_path, format="parquet")

    # Only read the columns we genuinely need
    wanted_cols = [
        "id_base",
        "paper_id",
        "title",
        "abstract",
        "categories",
        "year",
        "features",
    ]
    scanner = dataset.scanner(columns=wanted_cols, batch_size=batch_size)

    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]

    ids: list[str] = []
    paper_ids: list[str] = []
    titles: list[str] = []
    abstracts: list[str] = []
    categories: list = []
    years: list = []

    vocab_dim: int | None = None
    n_docs = 0

    print(
        f"[index-full] streaming batches (batch_size={batch_size}, "
        f"max_docs={max_docs if max_docs is not None else 'ALL'}) and building CSR components..."
    )

    for batch_idx, batch in enumerate(scanner.to_batches(), start=1):
        batch_len = batch.num_rows
        if batch_len == 0:
            continue

        ids_batch = batch["id_base"].to_pylist()
        pid_batch = batch["paper_id"].to_pylist()
        title_batch = batch["title"].to_pylist()
        abs_batch = batch["abstract"].to_pylist()
        cats_batch = batch["categories"].to_pylist()
        year_batch = batch["year"].to_pylist()

        features_py = batch["features"].to_pylist()

        for i, f in enumerate(features_py):
            # Stop early if we hit max_docs
            if max_docs is not None and n_docs >= max_docs:
                break

            # Metadata
            ids.append(str(ids_batch[i]))
            paper_ids.append(str(pid_batch[i]))
            titles.append(title_batch[i] or "")
            abstracts.append(abs_batch[i] or "")
            categories.append(cats_batch[i])
            years.append(year_batch[i])

            if f is None:
                indptr.append(indptr[-1])
                n_docs += 1
                continue

            if vocab_dim is None:
                vocab_dim = int(f["size"])

            idxs = f.get("indices") or []
            vals = f.get("values") or []

            if len(idxs) != len(vals):
                raise RuntimeError(
                    f"features.indices and features.values length mismatch at doc {n_docs}"
                )

            if idxs:
                idx_arr = np.asarray(idxs, dtype=np.int32)
                val_arr = np.asarray(vals, dtype=np.float32)
                indices.extend(idx_arr.tolist())
                data.extend(val_arr.tolist())

            indptr.append(indptr[-1] + len(idxs))
            n_docs += 1

        if max_docs is not None and n_docs >= max_docs:
            print(
                f"[index-full] reached max_docs={max_docs} at batch {batch_idx}, stopping early"
            )
            break

        # Progress every ~50k docs
        if n_docs % 50_000 < batch_len:
            print(f"[index-full] processed {n_docs} documents so far...")

    if vocab_dim is None:
        raise RuntimeError("No feature rows found in features_full; cannot build index")

    print(f"[index-full] building CSR matrix: num_docs={n_docs}, dim={vocab_dim}")

    data_arr = np.asarray(data, dtype=np.float32)
    indices_arr = np.asarray(indices, dtype=np.int32)
    indptr_arr = np.asarray(indptr, dtype=np.int64)

    mat = csr_matrix((data_arr, indices_arr, indptr_arr), shape=(n_docs, vocab_dim))

    csr_path = out_path / "full_index_csr.npz"
    ids_path = out_path / "full_index_ids.npy"
    pid_path = out_path / "full_index_paper_ids.npy"
    titles_path = out_path / "full_index_titles.npy"
    abs_path = out_path / "full_index_abstracts.npy"
    cats_path = out_path / "full_index_categories.npy"
    years_path = out_path / "full_index_years.npy"

    print(f"[index-full] saving CSR to {csr_path}")
    save_npz(csr_path, mat)

    print("[index-full] saving metadata arrays")
    np.save(ids_path, np.asarray(ids, dtype=object))
    np.save(pid_path, np.asarray(paper_ids, dtype=object))
    np.save(titles_path, np.asarray(titles, dtype=object))
    np.save(abs_path, np.asarray(abstracts, dtype=object))
    np.save(cats_path, np.asarray(categories, dtype=object))
    np.save(years_path, np.asarray(years, dtype=object))

    print(
        f"[index-full] done. Wrote index for {n_docs} documents to {out_path} "
        f"(dim={vocab_dim})"
    )


def main():
    build_full_index()


if __name__ == "__main__":
    main()
