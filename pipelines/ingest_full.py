from engine.data.ingestion import run_ingestion


def main():
    run_ingestion(
        input_path="data/raw/arxiv-metadata-oai-snapshot.json",
        output_path="data/processed/arxiv_full",
        partition_by="year",
        min_abstract_len=40,
        repartition=200,
    )


if __name__ == "__main__":
    main()
