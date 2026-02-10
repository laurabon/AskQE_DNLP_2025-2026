import json
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration
# Path reconstructed to be relative to the script location or absolute
PROJECT_ROOT = "/Users/laura/Desktop/BACKTRASLATION/AskQE_DNLP_2025-2026"
INPUT_FILE = os.path.join(PROJECT_ROOT, "biomqm/dev_with_backtranslation.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results Qwen3B baseline/backtranslation/dev_with_backtranslation_nllb.jsonl")
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# FLORES-200 Language Mapping for BioMQM
LANG_MAP = {
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "zh-CN": "zho_Hans",
    "en": "eng_Latn"
}

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    # Load input data
    print(f"Loading data from {INPUT_FILE}...")
    data = load_data(INPUT_FILE)
    if not data:
        print(f"Error: {INPUT_FILE} is empty or not found.")
        return

    print(f"Starting translation, saving to {OUTPUT_FILE}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Open file in write mode to start fresh
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, item in enumerate(data):
            lang_tgt = item.get("lang_tgt")
            
            if lang_tgt not in LANG_MAP:
                continue

            tgt_text = item.get("tgt")
            if not tgt_text or tgt_text == "{NS}":
                item["bt_tgt"] = ""
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                continue

            # Translation
            src_lang_code = LANG_MAP[lang_tgt]
            target_lang_code = LANG_MAP["en"]
            
            tokenizer.src_lang = src_lang_code
            inputs = tokenizer(tgt_text, return_tensors="pt").to(device)
            
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)

            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=128
                )
            
            bt_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            item["bt_tgt"] = bt_text
            
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()

            if i % 10 == 0:
                print(f"Progress: {i}/{len(data)}")

    print("Success!")

if __name__ == "__main__":
    main()
