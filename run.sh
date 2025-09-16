#!/bin/bash
set -e
python src/ingestion.py
python src/transformations.py
python src/streaming.py
python src/ml_pipeline.py
