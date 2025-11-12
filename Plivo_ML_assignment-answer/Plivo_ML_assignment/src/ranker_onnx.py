from typing import List, Tuple
import numpy as np

# Optional imports guarded to allow partial environments
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None

class PseudoLikelihoodRanker:
    def __init__(self, model_name: str = "distilbert-base-uncased", onnx_path: str = None, device: str = "cpu", max_length: int = 64):
        self.max_length = max_length
        self.model_name = model_name
        self.onnx = None
        self.torch_model = None
        self.device = device
        self.tokenizer = None
        if onnx_path and ort is not None:
            self._init_onnx(onnx_path)
        elif AutoTokenizer is not None and AutoModelForMaskedLM is not None:
            self._init_torch()
        else:
            raise RuntimeError("Neither onnxruntime nor transformers/torch are available. Please install requirements.")

    def _init_onnx(self, onnx_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.onnx = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    def _init_torch(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.torch_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.torch_model.eval()
        self.torch_model.to(self.device)

    def _batch_mask_positions(self, input_ids: np.ndarray, attn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create a batch of masked sequences, one for each non-[CLS]/[SEP] position
        mask_id = self.tokenizer.mask_token_id
        seq = input_ids[0]  # [1, L]
        L = int(attn[0].sum())
        positions = list(range(1, L-1))  # skip [CLS] and [SEP] equivalents
        batch = np.repeat(seq[None, :], len(positions), axis=0)
        for i, pos in enumerate(positions):
            batch[i, pos] = mask_id
        batch_attn = np.repeat(attn, len(positions), axis=0)
        return batch, batch_attn, np.array(positions, dtype=np.int64)

    def _score_with_onnx(self, text: str) -> float:
        # Tokenize to NumPy for ORT
        toks = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = toks["input_ids"]           # (1, L)
        attn = toks["attention_mask"]           # (1, L)

        # Figure out which token positions to score (skip special tokens)
        # Use attention mask to get real length L
        L = int(attn[0].sum())
        # Typically for HF models: position 0 = [CLS] or [BOS], position L-1 = [SEP] or [EOS]
        positions = list(range(1, L - 1))

        mask_id = self.tokenizer.mask_token_id
        seq = input_ids[0]                      # (L,)
        total = 0.0

        for pos in positions:
            # Make a masked copy with batch=1
            masked = seq.copy()
            orig_token_id = int(masked[pos])
            masked[pos] = mask_id

            ort_inputs = {
                "input_ids": masked[None, :].astype(np.int64),   # (1, L)
                "attention_mask": attn.astype(np.int64),         # (1, L)
            }

            # Run the model: logits shape (1, L, V)
            logits = self.onnx.run(None, ort_inputs)[0]
            logits_pos = logits[0, pos, :]                       # (V,)

            # log-softmax in a numerically stable way
            m = logits_pos.max()
            log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum())

            total += float(log_probs[orig_token_id])

        return total  # higher = better

    # def _score_with_onnx(self, text: str) -> float:
    #     toks = self.tokenizer(text, return_tensors="np", truncation=True, max_length=self.max_length)
    #     input_ids = toks["input_ids"]
    #     attn = toks["attention_mask"]
    #     batch, batch_attn, positions = self._batch_mask_positions(input_ids, attn)
    #     ort_inputs = {"input_ids": batch.astype(np.int64), "attention_mask": batch_attn.astype(np.int64)}
    #     logits = self.onnx.run(None, ort_inputs)[0]  # [B, L, V]
    #     # gather logprobs at the original token for each masked position
    #     orig = np.repeat(input_ids, len(positions), axis=0)
    #     rows = np.arange(len(positions))
    #     cols = positions
    #     token_ids = orig[rows, cols]
    #     # log softmax per row at the masked position
    #     logits_pos = logits[rows, cols, :]  # [B, V]
    #     m = logits_pos.max(axis=1, keepdims=True)
    #     log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum(axis=1, keepdims=True))
    #     picked = log_probs[np.arange(len(rows)), token_ids]
    #     return float(picked.sum())  # higher = better

    def _score_with_torch(self, text: str) -> float:
        import torch
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        # batch mask
        seq = input_ids[0]
        L = int(attn.sum())
        positions = list(range(1, L-1))
        batch = seq.unsqueeze(0).repeat(len(positions), 1)
        for i, pos in enumerate(positions):
            batch[i, pos] = self.tokenizer.mask_token_id
        batch_attn = attn.repeat(len(positions), 1)
        with torch.no_grad():
            out = self.torch_model(input_ids=batch, attention_mask=batch_attn).logits  # [B, L, V]
            orig = seq.unsqueeze(0).repeat(len(positions), 1)
            rows = torch.arange(len(positions))
            cols = torch.tensor(positions)
            token_ids = orig[rows, cols]
            logits_pos = out[rows, cols, :]
            log_probs = logits_pos.log_softmax(dim=-1)
            picked = log_probs[torch.arange(len(rows)), token_ids]
        return float(picked.sum().item())

    def score(self, sentences: List[str]) -> List[float]:
        return [self._score_with_onnx(s) if self.onnx is not None else self._score_with_torch(s) for s in sentences]

    def choose_best(self, candidates: List[str]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        scores = self.score(candidates)
        i = int(np.argmax(scores))
        return candidates[i]

import onnxruntime as ort
import numpy as np
from typing import List, Dict, Any
import re

class ONNXRanker:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = self._get_fast_tokenizer()
        
        # Optimization parameters
        self.max_length = 64
        self.valid_email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.valid_number_pattern = re.compile(r'^\+?[\d\s\-\(\)]{10,}$')

    def _get_fast_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except ImportError:
            # Fallback to basic tokenizer
            class BasicTokenizer:
                def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="np"):
                    # Simple whitespace tokenization for fallback
                    tokenized = []
                    for text in texts:
                        tokens = text.split()[:max_length]
                        token_ids = [hash(token) % 10000 for token in tokens]
                        # Pad to max_length
                        if len(token_ids) < max_length:
                            token_ids += [0] * (max_length - len(token_ids))
                        else:
                            token_ids = token_ids[:max_length]
                        tokenized.append(token_ids)
                    
                    return {
                        'input_ids': np.array(tokenized, dtype=np.int64),
                        'attention_mask': np.array([[1] * len(tokens) + [0] * (max_length - len(tokens)) 
                                                  for tokens in tokenized], dtype=np.int64)
                    }
            return BasicTokenizer()

    def score_candidates(self, original: str, candidates: List[str]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        # Early termination for perfect candidates
        perfect_candidates = []
        other_candidates = []
        
        for candidate in candidates:
            if self.is_perfect_candidate(original, candidate):
                perfect_candidates.append(candidate)
            else:
                other_candidates.append(candidate)
        
        # Return perfect candidates first
        if perfect_candidates:
            return [{"text": cand, "score": 1.0} for cand in perfect_candidates]
        
        # Short-circuit for very short texts
        if len(original.split()) <= 2:
            return [{"text": cand, "score": 0.9} for cand in candidates[:3]]
        
        # Limit candidates for faster processing
        if len(other_candidates) > 10:
            other_candidates = other_candidates[:10]
        
        scored_candidates = []
        for candidate in other_candidates:
            score = self.compute_pseudo_likelihood(candidate)
            scored_candidates.append({"text": candidate, "score": score})
        
        # Sort by score (higher is better)
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates

    def is_perfect_candidate(self, original: str, candidate: str) -> bool:
        """Check if candidate is perfect (contains valid email/number and matches pattern)"""
        # Check for valid email
        if self.valid_email_pattern.search(candidate) and not self.valid_email_pattern.search(original):
            return True
        
        # Check for valid number format
        if self.valid_number_pattern.search(candidate) and not self.valid_number_pattern.search(original):
            return True
        
        # Exact match with minor punctuation differences
        orig_clean = original.strip().lower()
        cand_clean = candidate.strip().lower()
        if orig_clean == cand_clean:
            return True
        
        return False

    def compute_pseudo_likelihood(self, text: str) -> float:
        """Compute pseudo-likelihood score using masked language modeling"""
        try:
            # Tokenize with length cap
            inputs = self.tokenizer(
                [text], 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors="np"
            )
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Run ONNX model
            outputs = self.session.run(
                None, 
                {
                    "input_ids": input_ids.astype(np.int64),
                    "attention_mask": attention_mask.astype(np.int64)
                }
            )
            
            logits = outputs[0]  # (1, seq_len, vocab_size)
            
            # Compute pseudo-likelihood with optimized masking
            total_score = 0.0
            num_masked = 0
            
            # Mask only content words (not special tokens)
            seq_len = np.sum(attention_mask[0]) - 2  # Exclude [CLS] and [SEP]
            if seq_len <= 0:
                return 0.0
            
            # Mask 30% of content tokens for efficiency
            mask_indices = list(range(1, min(seq_len + 1, self.max_length - 1)))
            if len(mask_indices) > 3:
                mask_indices = mask_indices[::3]  # Sample every 3rd token
            
            for pos in mask_indices:
                if pos >= len(input_ids[0]):
                    continue
                    
                original_id = input_ids[0, pos]
                if original_id in [0, 101, 102]:  # Skip padding and special tokens
                    continue
                
                # Mask this position
                masked_input = input_ids.copy()
                masked_input[0, pos] = 103  # [MASK] token
                
                # Get prediction
                mask_outputs = self.session.run(
                    None,
                    {
                        "input_ids": masked_input.astype(np.int64),
                        "attention_mask": attention_mask.astype(np.int64)
                    }
                )
                
                mask_logits = mask_outputs[0][0, pos]
                original_score = mask_logits[original_id]
                total_score += original_score
                num_masked += 1
            
            if num_masked == 0:
                return 0.0
                
            return float(total_score / num_masked)
            
        except Exception as e:
            print(f"Error in pseudo-likelihood computation: {e}")
            return 0.0

def load_onnx_ranker(model_path: str) -> ONNXRanker:
    return ONNXRanker(model_path)