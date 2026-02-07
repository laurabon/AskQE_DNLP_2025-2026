"""
BIOMQM QA Mapping Script (Index-based)
Combines source and bt answers to reconstruct the full dataset using row indexes.

Usage:
  python mapping_biomqm_index.py --pipeline vanilla --qa_dir /path/to/qa_results
"""

import json
import os
import argparse

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
SEVERITY_ORDER = {"Critical": 4, "Major": 3, "Minor": 2, "Neutral": 1}



def load_qa_map(file_path):
    """
    Loads QA file and builds a map: row_index -> {answers, questions}
    """
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
                questions = data.get('questions', [])
                row_indexes = data.get('row_indexes', [])
                
                for idx in row_indexes:
                    qa_map[idx] = {
                        'answers': answers,
                        'questions': questions
                    }
            except json.JSONDecodeError:
                continue
    return qa_map

def main():
    parser = argparse.ArgumentParser(description="BIOMQM QA Mapping Script (Index-based)")
    parser.add_argument("--pipeline", type=str, default="vanilla",
                        help="Pipeline name (default: vanilla)")
    parser.add_argument("--qa_dir", type=str, required=True,
                        help="Directory containing QA results")
    parser.add_argument("--qg_input_path", type=str,
                        help="Path to original dev_with_backtranslation.jsonl")
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Default QG path logic
    if args.qg_input_path:
        qg_file = args.qg_input_path
    else:
        # Try to find it in standard locations relative to repo root
        qg_file = os.path.join(project_root, "biomqm", "dev_with_backtranslation.jsonl")
        
    if not os.path.exists(qg_file):
        print(f"Original QG file not found at: {qg_file}")
        print("Please provide path via --qg_input_path")
        return

    # 1. Load Source QA Map
    if args.qa_dir:
        source_file = "/kaggle/input/baseline/source-vanilla.jsonl"
    else:
        source_file = "/kaggle/input/baseline/source-vanilla.jsonl"

    if not os.path.exists(source_file):
        print(f"Source answers file not found: {source_file}")
        print("Run QA with --mode source first!")
        return
        
    source_map = load_qa_map(source_file)
    print(f"Source Map size: {len(source_map)}")
    
    # 2. Load BT QA Map (all languages)
    # Using same directory as source file for consistency
    bt_map = {}
    for lang in LANGUAGES:
        bt_file = os.path.join("/kaggle/input/baseline", f"bt-{lang}-{args.pipeline}.jsonl")
        lang_map = load_qa_map(bt_file)
        bt_map.update(lang_map)
    print(f"BT Map size: {len(bt_map)}")

    # 3. Process Original File
    if args.qa_dir:
        output_file = os.path.join('/kaggle/working/askqe/extension-reasoning-qa', "QA", "biomqm", "mapped_baseline", f"all-{args.pipeline}.jsonl")
    else:
        output_file = os.path.join('/kaggle/working/askqe/extension-reasoning-qa', "QA", "biomqm", "mapped_baseline", f"all-{args.pipeline}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Processing original file: {qg_file}")
    
    rows_written = 0
    missing_source = 0
    missing_bt = 0
    
    with open(qg_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(f_in):
            try:
                original_row = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Lookup answers by Index
            src_qa_data = source_map.get(idx, {})
            bt_qa_data = bt_map.get(idx, {})
            
            answer_src = src_qa_data.get('answers', [])
            answer_bt = bt_qa_data.get('answers', [])
            
            # Questions: Prefer Source, then BT, then original (if exists)
            questions = src_qa_data.get('questions') or bt_qa_data.get('questions') or original_row.get('questions', [])
            
            if not answer_src: missing_source += 1
            if not answer_bt: missing_bt += 1
            
            # Build Output Row
            errors = original_row.get('errors_tgt', [])
            all_severities = [e.get('severity', 'Neutral') for e in errors]
            if not all_severities:
                all_severities = ["Neutral"]

            output_row = {
                'src': original_row.get('src', ''),
                'bt_tgt': original_row.get('bt_tgt', ''),
                'lang_tgt': original_row.get('lang_tgt', ''),
                'questions': questions,
                'answers_src': answer_src,  # Note naming requested: answers_src (plural)
                'answers_bt': answer_bt,    # Note naming requested: answers_bt (plural)
                'severities': all_severities, # Added list of all severities
                'docID': original_row.get('doc_id', ''), # User asked 'docID', mapping from 'doc_id'
                'system': original_row.get('system', '')
            }
            
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
            rows_written += 1
            
    print(f"\n{'='*50}")
    print(f"MAPPING COMPLETED (Index-based)")
    print(f"Output: {output_file}")
    print(f"Total rows written: {rows_written}")
    if missing_source: print(f"WARNING: {missing_source} rows missing Source answers")
    if missing_bt: print(f"WARNING: {missing_bt} rows missing BT answers")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()