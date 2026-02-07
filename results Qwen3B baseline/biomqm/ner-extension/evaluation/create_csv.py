"""
Generate CSV summary files from NER extension evaluation results.
Creates one CSV for SBERT and one for string-comparison with all languages.
"""

import json
import csv
import os

BASE_DIR = r"c:\Users\andos\DNLP-Project\askqe\results Qwen3B baseline\biomqm\ner-extension\evaluation"
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]

def load_jsonl(filepath):
    """Load JSONL file into a list of dicts."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def create_sbert_csv():
    """Create CSV from SBERT evaluation results."""
    output_path = os.path.join(BASE_DIR, "sbert_all_languages.csv")
    
    all_rows = []
    for lang in LANGUAGES:
        filepath = os.path.join(BASE_DIR, "sbert", f"{lang}.jsonl")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue
        
        data = load_jsonl(filepath)
        for row in data:
            all_rows.append({
                'lang': lang,
                'src': row.get('src', ''),
                'severity': row.get('severity', ''),
                'overall_similarity': row.get('overall_similarity', ''),
                'entity_scores': json.dumps(row.get('entity_scores', []))
            })
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['lang', 'src', 'severity', 'overall_similarity', 'entity_scores']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"SBERT CSV created: {output_path}")
    print(f"Total rows: {len(all_rows)}")

def create_string_comparison_csv():
    """Create CSV from string comparison evaluation results."""
    output_path = os.path.join(BASE_DIR, "string_comparison_all_languages.csv")
    
    all_rows = []
    for lang in LANGUAGES:
        filepath = os.path.join(BASE_DIR, "string-comparison", f"{lang}.jsonl")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue
        
        data = load_jsonl(filepath)
        for row in data:
            all_rows.append({
                'lang': lang,
                'src': row.get('src', ''),
                'severity': row.get('severity', ''),
                'overall_f1': row.get('overall_f1', ''),
                'overall_em': row.get('overall_em', ''),
                'entity_scores': json.dumps(row.get('entity_scores', []))
            })
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['lang', 'src', 'severity', 'overall_f1', 'overall_em', 'entity_scores']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"String Comparison CSV created: {output_path}")
    print(f"Total rows: {len(all_rows)}")

if __name__ == "__main__":
    create_sbert_csv()
    create_string_comparison_csv()
    print("\nDone!")
