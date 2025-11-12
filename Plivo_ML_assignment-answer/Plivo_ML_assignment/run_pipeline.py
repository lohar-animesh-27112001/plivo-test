import argparse, os
from src.postprocess_pipeline import run_file

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input", default="data/noisy_transcripts.jsonl")
#     ap.add_argument("--output", default="out/corrected.jsonl")
#     ap.add_argument("--names", default="data/names_lexicon.txt")
#     ap.add_argument("--onnx", default="models/distilbert-base-uncased.int8.onnx")
#     ap.add_argument("--device", default="cpu")
#     args = ap.parse_args()
#     os.makedirs(os.path.dirname(args.output), exist_ok=True)
#     run_file(args.input, args.output, args.names, onnx_model_path=args.onnx, device=args.device)

# if __name__ == "__main__":
#     main()

import json
import argparse
from typing import List, Dict, Any
from src.rules import correct_text
from src.ranker_onnx import load_onnx_ranker
import os
import time

def run_pipeline(
    input_path: str,
    output_path: str,
    names_lexicon_path: str,
    misspell_map_path: str,
    onnx_model_path: str,
    max_candidates: int = 15
):
    # Load data
    with open(input_path, 'r') as f:
        inputs = [json.loads(line) for line in f]
    
    # Initialize components
    ranker = load_onnx_ranker(onnx_model_path)
    
    results = []
    
    for i, item in enumerate(inputs):
        original_text = item.get("text", "")
        
        if not original_text.strip():
            results.append({"original": original_text, "corrected": original_text})
            continue
        
        # Stage 1: Generate candidates
        candidates = correct_text(original_text, names_lexicon_path, misspell_map_path)
        
        # Limit candidates for efficiency
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        
        # Stage 2: Rank candidates
        if len(candidates) == 1:
            # Skip ranking if only one candidate
            best_candidate = candidates[0]
        else:
            scored_candidates = ranker.score_candidates(original_text, candidates)
            if scored_candidates:
                best_candidate = scored_candidates[0]["text"]
            else:
                best_candidate = original_text
        
        results.append({
            "original": original_text,
            "corrected": best_candidate,
            "candidates_generated": len(candidates)
        })
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(inputs)} utterances")
    
    # Write results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Pipeline completed. Processed {len(inputs)} utterances.")
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/noisy_transcripts.jsonl")
    parser.add_argument("--output", default="out/corrected.jsonl")
    parser.add_argument("--names_lexicon", default="data/names_lexicon.txt")
    parser.add_argument("--misspell_map", default="data/misspell_map.json")
    parser.add_argument("--onnx_model", default="models/ranker_quantized.onnx")
    parser.add_argument("--max_candidates", type=int, default=15)
    
    args = parser.parse_args()
    
    start_time = time.time()
    run_pipeline(
        args.input,
        args.output,
        args.names_lexicon,
        args.misspell_map,
        args.onnx_model,
        args.max_candidates
    )
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")