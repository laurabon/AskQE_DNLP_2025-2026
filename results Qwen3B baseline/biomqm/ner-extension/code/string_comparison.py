"""
String Comparison Evaluation for NER Extension

Calculates F1, EM, etc. with entity-type breakdown.

Usage:
    python string_comparison.py --input_path /path/to/mapped.jsonl --output_dir /path/to/output/
"""

import json
import os
import argparse
from utils import compare_answers


LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def main():
    parser = argparse.ArgumentParser(description="String Comparison Evaluation for NER Extension")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to mapped JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing: {args.input_path}")
    
    # Stats structures
    results_by_lang = {lang: [] for lang in LANGUAGES}
    entity_stats = {}  # {entity_type: {total, f1_sum, em_sum}}
    severity_stats = {sev: [] for sev in ALL_SEVERITIES}
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            lang = row.get('lang_tgt', '')
            if lang not in LANGUAGES:
                continue
            
            entity_scores = row.get('entity_scores', {})
            severities = row.get('severities', ['Neutral'])
            
            # Calculate metrics per entity type
            row_entity_metrics = {}
            row_f1_sum = 0
            row_em_sum = 0
            valid_entities = 0
            
            for entity_type, scores in entity_scores.items():
                answer_src = scores.get('answer_src', '')
                answer_bt = scores.get('answer_bt', '')
                
                if not answer_src:
                    continue
                
                f1, em, chrf, bleu = compare_answers(answer_bt, answer_src)
                
                row_entity_metrics[entity_type] = {
                    'f1': f1,
                    'em': em,
                    'answer_src': answer_src,
                    'answer_bt': answer_bt
                }
                
                row_f1_sum += f1
                row_em_sum += em
                valid_entities += 1
                
                # Update entity stats
                if entity_type not in entity_stats:
                    entity_stats[entity_type] = {'total': 0, 'f1_sum': 0, 'em_sum': 0}
                entity_stats[entity_type]['total'] += 1
                entity_stats[entity_type]['f1_sum'] += f1
                entity_stats[entity_type]['em_sum'] += em
            
            # Calculate row average
            if valid_entities > 0:
                row_avg_f1 = row_f1_sum / valid_entities
                row_avg_em = row_em_sum / valid_entities
                
                # Add to severity stats
                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        severity_stats[sev].append((row_avg_f1, row_avg_em))
            else:
                row_avg_f1 = 0.0
                row_avg_em = 0.0
            
            # Build output row
            output_row = {
                'src': row.get('src', ''),
                'lang_tgt': lang,
                'severities': severities,
                'entity_metrics': row_entity_metrics,
                'overall_f1': row_avg_f1,
                'overall_em': row_avg_em
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
        print(f"Language: {lang}")
        print(f"Total rows: {len(rows)}")
        
        # Severity breakdown
        print(f"\n{'Severity':<10} {'Count':>6} {'F1':>8} {'EM':>8}")
        print("-" * 34)
        for sev in ALL_SEVERITIES:
            sev_data = [r for r in rows if sev in r['severities']]
            if sev_data:
                avg_f1 = sum(r['overall_f1'] for r in sev_data) / len(sev_data)
                avg_em = sum(r['overall_em'] for r in sev_data) / len(sev_data)
                print(f"{sev:<10} {len(sev_data):>6} {avg_f1:>8.3f} {avg_em:>8.3f}")
        
        print(f"\nSaved: {output_file}")
    
    # Print entity type summary
    print(f"\n{'='*60}")
    print("ENTITY TYPE SUMMARY (All Languages)")
    print(f"{'='*60}")
    print(f"{'Entity Type':<20} {'Count':>8} {'Avg F1':>10} {'Avg EM':>10}")
    print("-" * 50)
    for entity_type, stats in sorted(entity_stats.items()):
        if stats['total'] > 0:
            avg_f1 = stats['f1_sum'] / stats['total']
            avg_em = stats['em_sum'] / stats['total']
            print(f"{entity_type:<20} {stats['total']:>8} {avg_f1:>10.3f} {avg_em:>10.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
