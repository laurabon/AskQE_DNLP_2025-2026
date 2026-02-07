"""
SBERT Semantic Similarity Evaluation for BioMQM (Adapted for Index-Mapping & Unwind)

Questo script calcola la similarità semantica (Cosine Similarity) usando SBERT
tra le risposte generate dal SOURCE e quelle generate via BACKTRANSLATION.

Versione identica a string_comparison_biomqm.py per logica di I/O:
- Input: file unico mappa "all-vanilla.jsonl"
- Output: file JSONL per lingua contenente gli score
- Statistiche: Unwind delle severities (una frase conta per tutte le sue severities)
"""

import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ========================================
# SBERT MODEL SETUP
# ========================================
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_similarity(pred, ref):
    """Calcola Cosine Similarity tra due stringhe"""
    if not pred or not ref:
        return 0.0
        
    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
    encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        pred_output = model(**encoded_pred)
        ref_output = model(**encoded_ref)

    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
    pred_embed = F.normalize(pred_embed, p=2, dim=1)

    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
    ref_embed = F.normalize(ref_embed, p=2, dim=1)

    return F.cosine_similarity(pred_embed, ref_embed, dim=1).item()

# ========================================
# CONFIGURAZIONE
# ========================================

languages = ["de", "es", "fr", "ru", "zh-CN"]
pipeline = "vanilla"
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

# FILE PATHS (KAGGLE SPECIFIC)
mapped_file_path = "/kaggle/working/askqe/extension-reasoning-qa/QA/biomqm/mapped_baseline/all-vanilla.jsonl"
output_base_dir = "/kaggle/working/askqe/extension-reasoning-qa"


def main():
    if not os.path.exists(mapped_file_path):
        print(f"ERRORE: File mappato non trovato: {mapped_file_path}")
        return

    print(f"Caricamento dati da: {mapped_file_path}")
    
    results_by_lang = {lang: [] for lang in languages}
    
    # Struttura statistiche: stats[lang][severity] = list of cosine_scores
    stats = {lang: {sev: [] for sev in ALL_SEVERITIES} for lang in languages}

    # 1. ELABORAZIONE
    with open(mapped_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
                lang = row.get('lang_tgt')
                if lang not in languages: continue

                src_text = row.get('src', '')
                answers_src = row.get('answers_src', [])
                answers_bt = row.get('answers_bt', [])
                severities = row.get('severities', ["Neutral"])

                # Normalizzazione
                pred_list = [str(x) if x else "" for x in answers_bt]
                ref_list = [str(x) if x else "" for x in answers_src]

                # Padding / Troncamento
                len_p = len(pred_list)
                len_r = len(ref_list)
                
                if len_r == 0: continue
                
                if len_p < len_r:
                    pred_list.extend([""] * (len_r - len_p))
                elif len_p > len_r:
                    pred_list = pred_list[:len_r]

                # Calcolo Score
                row_scores = []
                row_sim_sum = 0
                valid_pairs = 0

                for pred, ref in zip(pred_list, ref_list):
                    if not ref.strip(): continue
                    
                    sim = get_similarity(pred, ref)
                    row_scores.append({"sbert_sim": sim})
                    
                    row_sim_sum += sim
                    valid_pairs += 1
                
                # Aggiornamento Statistiche (UNWIND)
                if valid_pairs > 0:
                    avg_sim_row = row_sim_sum / valid_pairs
                    
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats[lang][sev].append(avg_sim_row)

                # Output Row
                output_row = {
                    "src": src_text,
                    "severities": severities,
                    "scores": row_scores
                }
                
                results_by_lang[lang].append(output_row)

            except json.JSONDecodeError:
                print(f"Errore riga {i}")
                continue

    # 2. OUTPUT E REPORT
    for lang in languages:
        rows = results_by_lang[lang]
        if not rows:
            print(f"\nNessun dato per {lang}")
            continue

        # File output
        jsonl_output_file = os.path.join(
            output_base_dir, 
            "evaluation", 
            "sbert", 
            "biomqm", 
            "baseline",
            f"{lang}-{pipeline}.jsonl"
        )
        os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)

        with open(jsonl_output_file, 'w', encoding='utf-8') as out_f:
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report Unwind
        print(f"\n{'='*50}")
        print(f"SBERT Evaluation - Language: {lang}")
        print(f"Total Rows: {len(rows)}")
        print(f"{'='*50}")
        print(f"{'Severity':<10} {'Count':>6} {'Avg CosSim':>12}")
        print("-" * 30)

        for sev in ALL_SEVERITIES:
            scores_list = stats[lang][sev]
            count = len(scores_list)
            if count > 0:
                avg_val = sum(scores_list) / count
                print(f"{sev:<10} {count:>6} {avg_val:>12.3f}")
            else:
                print(f"{sev:<10} {count:>6} {'N/A':>12}")
        
        print(f"\nSaved: {jsonl_output_file}")


if __name__ == "__main__":
    main()