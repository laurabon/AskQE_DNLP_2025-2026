import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration
INPUT_FILE = "../biomqm/dev_with_backtranslation.jsonl"
OUTPUT_FILE = "../biomqm/dev_with_backtranslation_nllb.jsonl"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# FLORES-200 Language Mapping
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
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    # Load input data
    print(f"Loading data from {INPUT_FILE}...")
    data = load_data(INPUT_FILE)

    # Dictionary to hold results by language
    results = []

    for i, item in enumerate(data):
        lang_tgt = item.get("lang_tgt")
        if lang_tgt not in LANG_MAP:
            print(f"Skipping unknown language: {lang_tgt}")
            item["bt_tgt"] = ""
            results.append(item)
            continue

        tgt_text = item.get("tgt")
        if not tgt_text:
            item["bt_tgt"] = ""
            results.append(item)
            continue

        # Translation
        src_lang = LANG_MAP[lang_tgt]
        target_lang = LANG_MAP["en"]
        
        tokenizer.src_lang = src_lang
        inputs = tokenizer(tgt_text, return_tensors="pt").to(device)
        
        # We use convert_tokens_to_ids for broad compatibility
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128
            )
        
        bt_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        item["bt_tgt"] = bt_text
        results.append(item)

        if i % 50 == 0:
            print(f"Progress: {i}/{len(data)}")

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Success!")

if __name__ == "__main__":
    main()
