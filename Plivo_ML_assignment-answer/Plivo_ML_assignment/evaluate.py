import json
import argparse
from jiwer import wer, cer
from sklearn.metrics import f1_score
import re

def load_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def calculate_metrics(hypotheses, references):
    wer_score = wer(references, hypotheses)
    cer_score = cer(references, hypotheses)
    
    # Email accuracy
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    email_matches = 0
    total_emails = 0
    
    # Number accuracy
    number_pattern = re.compile(r'\b\d{6,}\b')
    number_matches = 0
    total_numbers = 0
    
    # Name F1 (simplified)
    name_pattern = re.compile(r'\b[A-Z][a-z]+\b')
    
    for hyp, ref in zip(hypotheses, references):
        # Email accuracy
        hyp_emails = email_pattern.findall(hyp)
        ref_emails = email_pattern.findall(ref)
        if ref_emails:
            total_emails += 1
            if hyp_emails and hyp_emails[0] == ref_emails[0]:
                email_matches += 1
        
        # Number accuracy
        hyp_numbers = number_pattern.findall(hyp)
        ref_numbers = number_pattern.findall(ref)
        if ref_numbers:
            total_numbers += 1
            if hyp_numbers and hyp_numbers[0] == ref_numbers[0]:
                number_matches += 1
    
    email_accuracy = email_matches / total_emails if total_emails > 0 else 0
    number_accuracy = number_matches / total_numbers if total_numbers > 0 else 0
    
    # Punctuation F1 (simplified)
    punct_chars = ['.', ',', '!', '?']
    hyp_punct = [1 if any(p in hyp for p in punct_chars) else 0 for hyp in hypotheses]
    ref_punct = [1 if any(p in ref for p in punct_chars) else 0 for ref in references]
    punct_f1 = f1_score(ref_punct, hyp_punct, zero_division=0)
    
    # Name F1 (simplified - check for proper nouns)
    hyp_names = [len(name_pattern.findall(hyp)) for hyp in hypotheses]
    ref_names = [len(name_pattern.findall(ref)) for ref in references]
    name_f1 = f1_score(ref_names, hyp_names, average='micro')
    
    return {
        'WER': wer_score,
        'CER': cer_score,
        'Email_Accuracy': email_accuracy,
        'Number_Accuracy': number_accuracy,
        'Punctuation_F1': punct_f1,
        'Name_F1': name_f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', required=True, help='Path to hypothesis file')
    parser.add_argument('--reference', required=True, help='Path to reference file')
    
    args = parser.parse_args()
    
    # Load data
    hyp_data = load_data(args.hypothesis)
    ref_data = load_data(args.reference)
    
    # Extract texts
    hypotheses = [item.get('corrected', '') for item in hyp_data]
    references = [item.get('text', '') for item in ref_data]
    
    # Calculate metrics
    metrics = calculate_metrics(hypotheses, references)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        if metric in ['WER', 'CER']:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value:.4f}")
    
    return metrics

if __name__ == '__main__':
    main()