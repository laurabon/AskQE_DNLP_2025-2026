"""
String Comparison Evaluation for BioMQM Dataset (Adapted for Index-Mapping)

Questo script calcola le metriche di similarità tra le risposte generate
dal SOURCE (inglese originale) e quelle generate dalla BACKTRANSLATION.
Le metriche sono: F1, Exact Match, chrF, BLEU.

Versione aggiornata per leggere direttamente il file unico mappato generato da 
mapping_biomqm_index.py, che contiene già answers_src e answers_bt allineati.
"""

import json
import nltk
import os
from utils import compare_answers

nltk.download("punkt")

# ========================================
# CONFIGURAZIONE BIOMQM
# ========================================

# Le 5 lingue presenti nel dataset BIOMQM
languages = ["de", "es", "fr", "ru", "zh-CN"]
pipelines = ["vanilla"]
# ========================================
# PERCORSI FILES (KAGGLE SPECIFIC)
# ========================================

# File unico mappato contenente TUTTI i dati (output di mapping_biomqm_index.py)
# path from step 268/275
mapped_file_path = "/kaggle/working/askqe/extension-reasoning-qa/QA/biomqm/mapped_baseline/all-vanilla.jsonl"

# Directory base per l'output dell'evaluation
output_base_dir = "/kaggle/working/askqe/extension-reasoning-qa"

# Le severity possibili per statistiche
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

# ========================================
# MAIN PROCESS
# ========================================

def main():
    if not os.path.exists(mapped_file_path):
        print(f"ERRORE: File mappato non trovato: {mapped_file_path}")
        print("Esegui prima mapping_biomqm_index.py!")
        return

    print(f"Caricamento dati da: {mapped_file_path}")
    
    # Dizionario per accumulare i risultati divisi per lingua e pipeline
    pipeline = "vanilla" 
    results_by_lang = {lang: [] for lang in languages}

    # Struttura per statistiche aggregate: stats[lang][severity] = list of (f1, em, bleu) tuples
    stats = {lang: {sev: [] for sev in ALL_SEVERITIES} for lang in languages}
    
    # 1. LETTURA E ELABORAZIONE (Singolo pass sul file)
    with open(mapped_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
                
                lang = row.get('lang_tgt')
                if lang not in languages:
                    continue 
                
                # Estrazione dati
                src_text = row.get('src', '')
                answers_src = row.get('answers_src', [])
                answers_bt = row.get('answers_bt', [])
                severities = row.get('severities', ["Neutral"])
                
                # === Normalizzazione ===
                pred_list = [str(x) if x else "" for x in answers_bt]
                ref_list = [str(x) if x else "" for x in answers_src]

                # === PADDING / TRUNCATING ===
                len_p = len(pred_list)
                len_r = len(ref_list)
                
                if len_r == 0:
                    continue
                    
                if len_p < len_r:
                    pred_list.extend([""] * (len_r - len_p))
                elif len_p > len_r:
                    pred_list = pred_list[:len_r]
                
                # === CALCOLO METRICHE ===
                row_scores = []
                # Accumulatori per media di RIGA
                row_f1_sum = 0
                row_em_sum = 0
                row_bleu_sum = 0
                valid_pairs = 0
                
                for pred, ref in zip(pred_list, ref_list):
                    if not ref.strip():
                        continue
                        
                    f1, EM, chrf, bleu = compare_answers(pred, ref)
                    row_scores.append({
                        "f1": f1,
                        "em": EM,
                        "chrf": chrf,
                        "bleu": bleu
                    })
                    
                    row_f1_sum += f1
                    row_em_sum += EM
                    row_bleu_sum += bleu
                    valid_pairs += 1
                
                # Calcola media aggragata per questa riga
                if valid_pairs > 0:
                    row_avg_f1 = row_f1_sum / valid_pairs
                    row_avg_em = row_em_sum / valid_pairs
                    row_avg_bleu = row_bleu_sum / valid_pairs
                    
                    # LOGICA UNWIND: Aggiungi questi score a TUTTE le severity presenti
                    for sev in severities:
                        # Normalizza severity nome (es. gestione errori typos se necessario, qui assumiamo corretto)
                        if sev in ALL_SEVERITIES:
                            stats[lang][sev].append((row_avg_f1, row_avg_em, row_avg_bleu))

                # === COSTRUZIONE RIGA OUTPUT ===
                output_row = {
                    "src": src_text,
                    "severities": severities, 
                    "scores": row_scores
                }
                
                results_by_lang[lang].append(output_row)
                
            except json.JSONDecodeError:
                print(f"Errore lettura riga {i}")
                continue

    # 2. SALVATAGGIO E REPORT
    for lang in languages:
        rows = results_by_lang[lang]
        if not rows:
            print(f"\nNessun dato trovato per {lang}")
            continue
            
        # Path output
        jsonl_output_file = os.path.join(
            output_base_dir, 
            "evaluation", 
            "string-comparison", 
            "biomqm", 
            "baseline",
            f"{lang}-{pipeline}.jsonl"
        )
        os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)
        
        # Scrittura su file
        with open(jsonl_output_file, 'w', encoding='utf-8') as out_f:
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        # Stampa Report con Unwind
        print(f"\n{'='*50}")
        print(f"Language: {lang} | Pipeline: {pipeline}")
        print(f"Total Rows: {len(rows)}")
        print(f"{'='*50}")
        
        # Stampa statistiche aggregate per severity (UNWOUND)
        print(f"{'Severity':<10} {'Count':>6} {'F1':>8} {'EM':>8} {'BLEU':>8}")
        print("-" * 44)
        
        for sev in ALL_SEVERITIES:
            scores_list = stats[lang][sev]
            count = len(scores_list)
            
            if count > 0:
                # Calcola media delle medie
                avg_f1 = sum(s[0] for s in scores_list) / count
                avg_em = sum(s[1] for s in scores_list) / count
                avg_bleu = sum(s[2] for s in scores_list) / count
                print(f"{sev:<10} {count:>6} {avg_f1:>8.3f} {avg_em:>8.3f} {avg_bleu:>8.3f}")
            else:
                print(f"{sev:<10} {count:>6} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

        print(f"\nSaved: {jsonl_output_file}")


if __name__ == "__main__":
    main()