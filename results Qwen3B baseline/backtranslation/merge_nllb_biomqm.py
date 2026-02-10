import json
import os

# Paths
PROJECT_ROOT = "/Users/laura/Desktop/BACKTRASLATION/AskQE_DNLP_2025-2026"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results Qwen3B baseline")
BT_DIR = os.path.join(RESULTS_DIR, "backtranslation")

BT_FILE = os.path.join(BT_DIR, "dev_with_backtranslation_nllb.jsonl")
QG_FILE = os.path.join(RESULTS_DIR, "QG", "biomqm", "vanilla_qwen-3b.jsonl")
MERGED_OUTPUT = os.path.join(BT_DIR, "nllb_qg_merged.jsonl")

def load_qg_data(file_path):
    """Loads QG data and keys it by the source sentence."""
    data = {}
    if not os.path.exists(file_path):
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            # BioMQM QG uses "src" for the English sentence
            if "src" in item:
                data[item["src"]] = item
            elif "en" in item:
                data[item["en"]] = item
    return data

def main():
    print(f"Loading QG data from {QG_FILE}...")
    qg_lookup = load_qg_data(QG_FILE)
    print(f"Loaded {len(qg_lookup)} unique source sentences with questions.")

    if not os.path.exists(BT_FILE):
        print(f"Error: {BT_FILE} not found. Run backtranslate_nllb.py first.")
        return

    print(f"Loading Backtranslated data from {BT_FILE} and merging...")
    merged_count = 0
    missing_count = 0

    with open(MERGED_OUTPUT, "w", encoding="utf-8") as out_f:
        with open(BT_FILE, "r", encoding="utf-8") as in_f:
            for line in in_f:
                bt_item = json.loads(line.strip())
                src_text = bt_item["src"]
                
                if src_text in qg_lookup:
                    qg_item = qg_lookup[src_text]
                    # Create the row format expected by the QA script
                    merged_item = {
                        "doc_id": qg_item.get("doc_id", "unknown"),
                        "src": src_text,
                        "lang_tgt": bt_item["lang_tgt"],
                        "bt_tgt": bt_item["bt_tgt"],
                        "questions": qg_item["questions"],
                        "atomic_facts": qg_item.get("atomic_facts", ""),
                        "semantic_roles": qg_item.get("semantic_roles", "")
                    }
                    out_f.write(json.dumps(merged_item, ensure_ascii=False) + "\n")
                    merged_count += 1
                else:
                    missing_count += 1
                    if missing_count < 5:
                        print(f"Warning: No match for sentence starting with: {src_text[:50]}...")

    print(f"Success! Merged {merged_count} items. {missing_count} items skipped (no QG match).")
    print(f"Output saved to {MERGED_OUTPUT}")

if __name__ == "__main__":
    main()
