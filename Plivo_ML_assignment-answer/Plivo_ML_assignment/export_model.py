import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import onnxruntime as ort
import os

def export_model():
    print("Loading DistilBERT model...")
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randint(0, 1000, (1, 64))
    
    torch.onnx.export(
        model,
        dummy_input,
        "models/ranker.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14
    )
    
    # Quantize model (simplified approach)
    print("Quantizing model...")
    try:
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(
            "models/ranker.onnx",
            "models/ranker_quantized.onnx"
        )
        print("Model quantized successfully!")
    except Exception as e:
        print(f"Quantization failed: {e}")
        # Copy original as fallback
        import shutil
        shutil.copy("models/ranker.onnx", "models/ranker_quantized.onnx")
        print("Using original model as fallback")
    
    print("Model export completed!")

if __name__ == "__main__":
    export_model()