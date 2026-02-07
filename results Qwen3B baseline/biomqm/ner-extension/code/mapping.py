"""
Mapping Script for NER Extension

Combines source and BT answers using row indexes, with entity-level breakdown.

Usage:
    python mapping.py --qa_source_path /path/to/qa_source.jsonl \
                      --qa_bt_dir /path/to/bt_files/ \
                      --bt_original_path /path/to/dev_with_backtranslation.jsonl \
                      --output_path /path/to/mapped.jsonl
"""

import json
import os
import argparse


LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]


def load_qa_map(file_path):
    """Load QA file and build map: row_index -> answers list."""
    qa_map = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return qa_map
    
    print(f"Loading {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                answers = data.get('answers', [])
                row_indexes = data.get('row_indexes', [])
                
                for idx in row_indexes:
                    qa_map[idx] = answers
            except json.JSONDecodeError:
                continue
    
    return qa_map


def calculate_entity_scores(answers_src, answers_bt):
    """
    Calculate per-entity type scores.
    
    Returns:
        dict: {entity_type: {answer_src, answer_bt, match: bool}}
    """
    entity_scores = {}
    
    # Match answers by entity_type
    src_by_type = {}
    bt_by_type = {}
    
    for a in answers_src:
        entity_type = a.get('entity_type', 'UNKNOWN')
        if entity_type not in src_by_type:
            src_by_type[entity_type] = []
        src_by_type[entity_type].append(a.get('answer', ''))
    
    for a in answers_bt:
        entity_type = a.get('entity_type', 'UNKNOWN')
        if entity_type not in bt_by_type:
            bt_by_type[entity_type] = []
        bt_by_type[entity_type].append(a.get('answer', ''))
    
    # Calculate scores per entity type
    all_types = set(src_by_type.keys()) | set(bt_by_type.keys())
    
    for entity_type in all_types:
        src_answers = src_by_type.get(entity_type, [''])
        bt_answers = bt_by_type.get(entity_type, [''])
        
        # Simple match: compare first answer of each type
        src_ans = src_answers[0] if src_answers else ''
        bt_ans = bt_answers[0] if bt_answers else ''
        
        # Exact match check (case insensitive)
        match = src_ans.lower().strip() == bt_ans.lower().strip() if src_ans and bt_ans else False
        
        entity_scores[entity_type] = {
            'answer_src': src_ans,
            'answer_bt': bt_ans,
            'match': match
        }
    
    return entity_scores


def main():
    parser = argparse.ArgumentParser(description="Mapping for NER Extension")
    parser.add_argument("--qa_source_path", type=str, required=True,
                        help="Path to source QA JSONL file")
    parser.add_argument("--qa_bt_dir", type=str, required=True,
                        help="Directory containing BT QA files (bt-{lang}.jsonl)")
    parser.add_argument("--bt_original_path", type=str, required=True,
                        help="Path to original dev_with_backtranslation.jsonl")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output mapped JSONL file")
    args = parser.parse_args()
    
    if not os.path.exists(args.qa_source_path):
        print(f"Error: Source QA file not found: {args.qa_source_path}")
        return
    
    if not os.path.exists(args.bt_original_path):
        print(f"Error: Original BT file not found: {args.bt_original_path}")
        return
    
    # Load source QA map
    source_map = load_qa_map(args.qa_source_path)
    print(f"Source map size: {len(source_map)}")
    
    # Load BT QA maps (all languages)
    bt_map = {}
    for lang in LANGUAGES:
        bt_file = os.path.join(args.qa_bt_dir, f"bt-{lang}.jsonl")
        if os.path.exists(bt_file):
            lang_map = load_qa_map(bt_file)
            bt_map.update(lang_map)
    print(f"BT map size: {len(bt_map)}")
    
    # Process original file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"Processing: {args.bt_original_path}")
    
    rows_written = 0
    entity_type_stats = {}  # Track entity type occurrences
    
    with open(args.bt_original_path, 'r', encoding='utf-8') as f_in, \
         open(args.output_path, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(f_in):
            try:
                original_row = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Get answers
            answers_src = source_map.get(idx, [])
            answers_bt = bt_map.get(idx, [])
            
            # Calculate entity scores
            entity_scores = calculate_entity_scores(answers_src, answers_bt)
            
            # Track stats
            for entity_type, scores in entity_scores.items():
                if entity_type not in entity_type_stats:
                    entity_type_stats[entity_type] = {'total': 0, 'matches': 0}
                entity_type_stats[entity_type]['total'] += 1
                if scores['match']:
                    entity_type_stats[entity_type]['matches'] += 1
            
            # Get severities from errors
            errors = original_row.get('errors_tgt', [])
            severities = [e.get('severity', 'Neutral') for e in errors]
            if not severities:
                severities = ["Neutral"]
            
            # Build output row
            output_row = {
                'src': original_row.get('src', ''),
                'bt_tgt': original_row.get('bt_tgt', ''),
                'lang_tgt': original_row.get('lang_tgt', ''),
                'entity_scores': entity_scores,
                'answers_src': answers_src,
                'answers_bt': answers_bt,
                'severities': severities,
                'doc_id': original_row.get('doc_id', ''),
                'system': original_row.get('system', '')
            }
            
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
            rows_written += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MAPPING COMPLETE")
    print(f"Output: {args.output_path}")
    print(f"Total rows: {rows_written}")
    print(f"\nEntity Type Statistics:")
    print(f"{'Type':<20} {'Total':>8} {'Matches':>8} {'Match%':>8}")
    print("-" * 46)
    for entity_type, stats in sorted(entity_type_stats.items()):
        pct = (stats['matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{entity_type:<20} {stats['total']:>8} {stats['matches']:>8} {pct:>7.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
