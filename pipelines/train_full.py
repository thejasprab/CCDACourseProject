# pipelines/train_full.py
from engine.ml.train import train_model


def main():
    """
    Train a TF-IDF model on the FULL ingested dataset.

    Settings are tuned to be lighter than the sample model to avoid OOM:
      - smaller vocab_size
      - higher min_df
      - extra_stopwords disabled (heavy aggregation over full corpus)
    """
    train_model(
        input_parquet="data/processed/arxiv_full",
        model_dir="data/models/tfidf_full",
        features_out="data/processed/features_full",
        vocab_size=120_000,       # was 250_000
        min_df=10,                # was 5
        use_bigrams=False,
        extra_stopwords_topdf=0,  # disable extra stopword computation
    )


if __name__ == "__main__":
    main()
