import time
import json
import argparse
from src.rules import correct_text
from src.ranker_onnx import load_onnx_ranker
import numpy as np

def measure_latency(
    input_path: str,
    names_lexicon_path: str,
    misspell_map_path: str,
    onnx_model_path: str,
    num_runs: int = 100,
    num_warmup: int = 10
):
    # Load data
    with open(input_path, 'r') as f:
        inputs = [json.loads(line) for line in f]
    
    # Initialize components
    ranker = load_onnx_ranker(onnx_model_path)
    
    latencies = []
    
    # Warmup runs
    print("Running warmup...")
    for i in range(num_warmup):
        item = inputs[i % len(inputs)]
        original_text = item.get("text", "")
        
        candidates = correct_text(original_text, names_lexicon_path, misspell_map_path)
        if len(candidates) > 1:
            ranker.score_candidates(original_text, candidates)
    
    # Actual measurement
    print("Measuring latency...")
    for i in range(num_runs):
        item = inputs[i % len(inputs)]
        original_text = item.get("text", "")
        
        start_time = time.perf_counter()
        
        # Full pipeline
        candidates = correct_text(original_text, names_lexicon_path, misspell_map_path)
        if len(candidates) > 1:
            ranked = ranker.score_candidates(original_text, candidates)
        else:
            ranked = [{"text": candidates[0], "score": 1.0}]
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    # Calculate statistics
    latencies_sorted = sorted(latencies)
    p50 = np.percentile(latencies_sorted, 50)
    p95 = np.percentile(latencies_sorted, 95)
    
    print(f"\n=== Latency Results ===")
    print(f"Number of runs: {num_runs}")
    print(f"Warmup runs: {num_warmup}")
    print(f"P50 latency: {p50:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")
    print(f"Min latency: {min(latencies):.2f} ms")
    print(f"Max latency: {max(latencies):.2f} ms")
    
    return p50, p95

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/noisy_transcripts.jsonl")
    parser.add_argument("--names_lexicon", default="data/names_lexicon.txt")
    parser.add_argument("--misspell_map", default="data/misspell_map.json")
    parser.add_argument("--onnx_model", default="models/ranker_quantized.onnx")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=10)
    
    args = parser.parse_args()
    
    p50, p95 = measure_latency(
        args.input,
        args.names_lexicon,
        args.misspell_map,
        args.onnx_model,
        args.num_runs,
        args.num_warmup
    )
    
    # Check if latency meets requirements
    if p95 <= 30:
        print("✅ SUCCESS: P95 latency meets requirement (≤ 30 ms)")
    else:
        print("❌ FAILED: P95 latency exceeds requirement (≤ 30 ms)")