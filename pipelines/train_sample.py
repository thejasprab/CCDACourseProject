from engine.ml.train import train_model


def main():
    train_model(
        input_parquet="data/processed/arxiv_sample",
        model_dir="data/models/tfidf_sample",
        features_out="data/processed/features_sample",
        vocab_size=80000,
        min_df=3,
        use_bigrams=False,
        extra_stopwords_topdf=200,
    )


if __name__ == "__main__":
    main()
