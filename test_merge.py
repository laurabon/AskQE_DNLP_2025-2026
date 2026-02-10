import json
import os
from collections import Counter

PROJECT_ROOT = '/Users/laura/Desktop/BACKTRASLATION/AskQE_DNLP_2025-2026'
NLLB_FILE = os.path.join(PROJECT_ROOT, 'results Qwen3B baseline', 'backtranslation', 'dev_with_backtranslation_nllb.jsonl')
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, 'results Qwen3B baseline', 'QG', 'biomqm', 'vanilla_qwen-3b.jsonl')

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("Loading files...")
nllb_data = load_jsonl(NLLB_FILE)
questions_data = load_jsonl(QUESTIONS_FILE)

print(f"NLLB records: {len(nllb_data)}")
print(f"Question records: {len(questions_data)}")

# Analyze languages in NLLB
languages = Counter(item.get('lang_tgt') for item in nllb_data)
print(f"Languages in NLLB file: {languages}")

# Map questions
print("Mapping questions...")
questions_map = {item['src']: item['questions'] for item in questions_data if 'src' in item}

merged_count = 0
mismatch_count = 0

for item in nllb_data:
    src = item.get('src')
    if src in questions_map:
        merged_count += 1
    else:
        mismatch_count += 1

print(f"Merged successfully: {merged_count}")
print(f"Mismatches: {mismatch_count}")

if mismatch_count == 0:
    print("SUCCESS: All NLLB records have matching questions.")
else:
    print("WARNING: Some records could not be merged.")
