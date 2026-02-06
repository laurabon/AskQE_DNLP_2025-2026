"""
BioMQM Mapping Script for Reasoning Extension

Combines source and bt answers from unique files to reconstruct
all rows with their original row positions.

Usage:
    python mapping_biomqm.py --pipeline vanilla
"""

import json
import os
import argparse


LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]


def main():
    parser = argparse.ArgumentParser(description="BioMQM Mapping for Reasoning Extension")
    parser.add_argument("--pipeline", type=str, default="vanilla", help="Pipeline name")
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(extension_dir)
    
    results_dir = os.path.join(extension_dir, "results", "biomqm")
    unique_dir = os.path.join(results_dir, "unique")
    mapped_dir = os.path.join(results_dir, "mapped")
    os.makedirs(mapped_dir, exist_ok=True)
    
    # Load source answers (keyed by src)
    source_file = os.path.join(unique_dir, f"source-{args.pipeline}.jsonl")
    source_answers = {}
    
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return
    
    print(f"Loading source answers from: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            src = data.get('src', '')
            source_answers[src] = data.get('answers', '')
    
    print(f"  Loaded {len(source_answers)} unique source answers")
    
    # Process each language
    for lang in LANGUAGES:
        bt_file = os.path.join(unique_dir, f"bt-{lang}-{args.pipeline}.jsonl")
        
        if not os.path.exists(bt_file):
            print(f"⚠️  BT file not found for {lang}, skipping")
            continue
        
        print(f"\nProcessing {lang}...")
        
        # Load bt answers (keyed by (src, bt_tgt))
        bt_data_list = []
        with open(bt_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                bt_data_list.append(data)
        
        print(f"  Loaded {len(bt_data_list)} unique BT entries")
        
        # Reconstruct all rows
        output_file = os.path.join(mapped_dir, f"{lang}-{args.pipeline}.jsonl")
        total_rows = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for bt_data in bt_data_list:
                src = bt_data.get('src', '')
                bt_tgt = bt_data.get('bt_tgt', '')
                questions = bt_data.get('questions', '')
                bt_answers = bt_data.get('answers', '')
                row_indexes = bt_data.get('row_indexes', [])
                
                # Get source answers
                src_answers = source_answers.get(src, '')
                
                # Write one row per original index
                for row_idx in row_indexes:
                    output_row = {
                        'original_row_index': row_idx,
                        'src': src,
                        'bt_tgt': bt_tgt,
                        'lang_tgt': lang,
                        'questions': questions,
                        'source_answers': src_answers,
                        'bt_answers': bt_answers
                    }
                    f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
                    total_rows += 1
        
        print(f"  ✅ Wrote {total_rows} rows to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("MAPPING COMPLETE")
    print("=" * 60)
    
    total_all = 0
    for lang in LANGUAGES:
        output_file = os.path.join(mapped_dir, f"{lang}-{args.pipeline}.jsonl")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                count = sum(1 for _ in f)
            print(f"{lang}: {count} rows")
            total_all += count
    
    print(f"\nTOTAL: {total_all} rows")


if __name__ == "__main__":
    main()
