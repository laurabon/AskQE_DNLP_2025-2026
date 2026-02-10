import json
import os

PROJECT_ROOT = '/Users/laura/Desktop/BACKTRASLATION/AskQE_DNLP_2025-2026'
NLLB_FILE = os.path.join(PROJECT_ROOT, 'results Qwen3B baseline', 'backtranslation', 'dev_with_backtranslation_nllb.jsonl')
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, 'results Qwen3B baseline', 'QG', 'biomqm', 'vanilla_qwen-3b.jsonl')

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

nllb_data = load_jsonl(NLLB_FILE)
questions_data = load_jsonl(QUESTIONS_FILE)

print("NLLB First Src:", repr(nllb_data[0].get('src')))
print("Questions First Src:", repr(questions_data[0].get('src')))

print("NLLB Second Src:", repr(nllb_data[1].get('src')))
print("Questions Second Src:", repr(questions_data[1].get('src')))
