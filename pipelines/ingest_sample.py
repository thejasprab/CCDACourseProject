from engine.data.ingestion import run_ingestion


def main():
    run_ingestion(
        input_path="data/sample/arxiv-sample.jsonl",
        output_path="data/processed/arxiv_sample",
        partition_by="year",
        min_abstract_len=40,
        repartition=64,
    )


if __name__ == "__main__":
    main()
