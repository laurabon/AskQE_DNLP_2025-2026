import json
from transformers import MarianMTModel, MarianTokenizer
import os

# Percorso del toy dataset (cartella corretta)
input_dir = "results Qwen3B baseline/toy_data"  # La cartella contenente il tuo toy dataset
output_dir = "results Qwen3B baseline/backtranslation"  # La cartella dove verrà salvato l'output

# Creiamo la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Carica il modello e il tokenizer
model_name = "facebook/nllb-200-distilled-1.3B"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Funzione di traduzione
def translate(text, source_lang, target_lang, model, tokenizer):
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang

    # Tokenizza il testo di input
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Esegui la traduzione
    translated = model.generate(**tokens)
    
    # Decodifica il testo tradotto
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Perturbazioni per cui eseguire la traduzione (solo per una lingua)
perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]

for perturbation in perturbations:
    # Percorso del file di input (toy dataset)
    input_file = os.path.join(input_dir, f"toy_{perturbation}.jsonl")
    # Percorso del file di output
    output_file = os.path.join(output_dir, f"bt_{perturbation}_en.jsonl")

    updated_jsonl = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            pert_key = "pert_es"  # Supponiamo che il dataset sia in spagnolo (o un'altra lingua specifica)

            if pert_key in data:
                print(f"Perturbed translation (es): ", data[pert_key])
                try:
                    # Traduci usando il modello NLLB
                    translated_text = translate(data[pert_key], source_lang="es", target_lang="en", model=model, tokenizer=tokenizer)
                    print("Backtranslation: ", translated_text)
                    data[f"bt_{pert_key}"] = translated_text
                except Exception as e:
                    print(f"Translation failed for: {data[pert_key]} with error: {e}")
                    data[f"bt_{pert_key}"] = ""
            updated_jsonl.append(data)

    # Salva l'output tradotto
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in updated_jsonl:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Traduzione completata per {perturbation}. Output salvato in {output_file}")


