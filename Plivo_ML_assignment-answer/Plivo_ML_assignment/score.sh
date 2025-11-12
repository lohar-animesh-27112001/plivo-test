#!/bin/bash

set -e

echo "Setting up environment..."

# Create directories
mkdir -p models
mkdir -p out

# Install dependencies if needed
# pip install -r requirements.txt

echo "Exporting and quantizing model..."

# Export model to ONNX and quantize (if not already done)
python export_model.py

echo "Running pipeline..."

# Run the improved pipeline
python run_pipeline.py \
    --input data/noisy_transcripts.jsonl \
    --output out/corrected.jsonl \
    --names_lexicon data/names_lexicon.txt \
    --misspell_map data/misspell_map.json \
    --onnx_model models/ranker_quantized.onnx \
    --max_candidates 15

echo "Evaluating results..."

# Evaluate metrics
python evaluate.py \
    --hypothesis out/corrected.jsonl \
    --reference data/gold.jsonl

echo "Measuring latency..."

# Measure latency
python measure_latency.py \
    --input data/noisy_transcripts.jsonl \
    --names_lexicon data/names_lexicon.txt \
    --misspell_map data/misspell_map.json \
    --onnx_model models/ranker_quantized.onnx

echo "All tasks completed!"