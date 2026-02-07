"""
SBERT Semantic Similarity Evaluation for NER Extension

Calculates cosine similarity with entity-type breakdown.

Usage:
    python sbert.py --input_path /path/to/mapped.jsonl --output_dir /path/to/output/
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def load_sbert_model():
    """Load SBERT model and tokenizer."""
    print(f"Loading SBERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_similarity(tokenizer, model, text1, text2):
    """Calculate cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    encoded1 = tokenizer(text1, padding=True, truncation=True, return_tensors='pt')
    encoded2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        output1 = model(**encoded1)
        output2 = model(**encoded2)
    
    embed1 = mean_pooling(output1, encoded1['attention_mask'])
    embed1 = F.normalize(embed1, p=2, dim=1)
    
    embed2 = mean_pooling(output2, encoded2['attention_mask'])
    embed2 = F.normalize(embed2, p=2, dim=1)
    
    return F.cosine_similarity(embed1, embed2, dim=1).item()


def main():
    parser = argparse.ArgumentParser(description="SBERT Evaluation for NER Extension")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to mapped JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    tokenizer, model = load_sbert_model()
    
    print(f"Processing: {args.input_path}")
    
    # Stats structures
    results_by_lang = {lang: [] for lang in LANGUAGES}
    entity_stats = {}  # {entity_type: {total, sim_sum}}
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if i % 100 == 0:
                print(f"Processing row {i}...")
            
            lang = row.get('lang_tgt', '')
            if lang not in LANGUAGES:
                continue
            
            entity_scores = row.get('entity_scores', {})
            severities = row.get('severities', ['Neutral'])
            
            # Calculate similarity per entity type
            row_entity_metrics = {}
            row_sim_sum = 0
            valid_entities = 0
            
            for entity_type, scores in entity_scores.items():
                answer_src = scores.get('answer_src', '')
                answer_bt = scores.get('answer_bt', '')
                
                if not answer_src:
                    continue
                
                sim = get_similarity(tokenizer, model, answer_src, answer_bt)
                
                row_entity_metrics[entity_type] = {
                    'similarity': sim,
                    'answer_src': answer_src,
                    'answer_bt': answer_bt
                }
                
                row_sim_sum += sim
                valid_entities += 1
                
                # Update entity stats
                if entity_type not in entity_stats:
                    entity_stats[entity_type] = {'total': 0, 'sim_sum': 0}
                entity_stats[entity_type]['total'] += 1
                entity_stats[entity_type]['sim_sum'] += sim
            
            # Calculate row average
            row_avg_sim = row_sim_sum / valid_entities if valid_entities > 0 else 0.0
            
            # Build output row
            output_row = {
                'src': row.get('src', ''),
                'lang_tgt': lang,
                'severities': severities,
                'entity_metrics': row_entity_metrics,
                'overall_similarity': row_avg_sim
            }
            
            results_by_lang[lang].append(output_row)
    
    # Save results per language
    for lang in LANGUAGES:
        rows = results_by_lang[lang]
        if not rows:
            continue
        
        output_file = os.path.join(args.output_dir, f"{lang}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"SBERT - Language: {lang}")
        print(f"Total rows: {len(rows)}")
        
        # Severity breakdown
        print(f"\n{'Severity':<10} {'Count':>6} {'Avg Sim':>10}")
        print("-" * 28)
        for sev in ALL_SEVERITIES:
            sev_data = [r for r in rows if sev in r['severities']]
            if sev_data:
                avg_sim = sum(r['overall_similarity'] for r in sev_data) / len(sev_data)
                print(f"{sev:<10} {len(sev_data):>6} {avg_sim:>10.3f}")
        
        print(f"\nSaved: {output_file}")
    
    # Print entity type summary
    print(f"\n{'='*60}")
    print("ENTITY TYPE SUMMARY (SBERT - All Languages)")
    print(f"{'='*60}")
    print(f"{'Entity Type':<20} {'Count':>8} {'Avg Similarity':>15}")
    print("-" * 45)
    for entity_type, stats in sorted(entity_stats.items()):
        if stats['total'] > 0:
            avg_sim = stats['sim_sum'] / stats['total']
            print(f"{entity_type:<20} {stats['total']:>8} {avg_sim:>15.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
