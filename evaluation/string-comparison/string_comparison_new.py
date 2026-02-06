import json
import nltk
import os
from utils import compare_answers

nltk.download("punkt")

languages = ["es", "fr", "hi", "tl", "zh"]
pipelines = ["atomic", "semantic", "vanilla"]

perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact",
                 "intensifier", "expansion_impact", "omission", "alteration"]

# Use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
project_root = os.path.dirname(os.path.dirname(script_dir))
# Se necessario, forza il percorso assoluto:
results_dir = "/content/askqe/results Qwen3B baseline"

for language in languages:
    for pipeline in pipelines:
        for perturbation in perturbations:
            predicted_file = os.path.join(results_dir, "QA_FIXED", "bt", f"{language}", f"{pipeline}", f"{language}-{pipeline}-{perturbation}.jsonl")
            reference_file = os.path.join(results_dir, "QA_FIXED", "source", f"en-{pipeline}.jsonl")

            results_list = []
            
            # Se manca il file, salta
            if not os.path.exists(predicted_file) or not os.path.exists(reference_file):
                # print(f"File mancante: {language}-{pipeline}-{perturbation}") # Decommenta se vuoi debug
                continue

            try:
                with open(predicted_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file:
                    for pred_line, ref_line in zip(pred_file, ref_file):
                        try:
                            pred_data = json.loads(pred_line)
                            ref_data = json.loads(ref_line)

                            predicted_answers = pred_data.get("answers", [])
                            reference_answers = ref_data.get("answers", [])

                            # Safety Checks: converte stringhe in liste se necessario
                            if isinstance(predicted_answers, str):
                                try: predicted_answers = json.loads(predicted_answers)
                                except: predicted_answers = []
                            if isinstance(reference_answers, str):
                                try: reference_answers = json.loads(reference_answers)
                                except: reference_answers = []

                            if not isinstance(predicted_answers, list): predicted_answers = []
                            if not isinstance(reference_answers, list): reference_answers = []

                            # === LOGICA CORRETTA (PADDING & TRUNCATING) ===
                            len_p = len(predicted_answers)
                            len_r = len(reference_answers)

                            # 1. Se il SOURCE non ha risposte, saltiamo la riga (non c'è nulla da valutare)
                            if len_r == 0:
                                continue

                            # 2. Se il BT ha meno risposte, aggiungiamo stringhe vuote ("")
                            if len_p < len_r:
                                predicted_answers.extend([""] * (len_r - len_p))
                            
                            # 3. Se il BT ha più risposte, tagliamo l'eccesso
                            elif len_p > len_r:
                                predicted_answers = predicted_answers[:len_r]
                            
                            # Ora len_p == len_r, possiamo procedere al confronto

                            row_scores = []
                            for pred, ref in zip(predicted_answers, reference_answers):
                                # Convert non-string values to strings
                                if not isinstance(pred, str):
                                    pred = str(pred) if pred is not None else ""
                                if not isinstance(ref, str):
                                    ref = str(ref) if ref is not None else ""
                                
                                # IMPORTANTE: Saltiamo SOLO se la REFERENCE è vuota.
                                # Se 'pred' è vuoto (il nostro padding), deve passare per prendere 0.
                                if not ref.strip():
                                    continue
                                
                                f1, EM, chrf, bleu = compare_answers(pred, ref)
                                row_scores.append({
                                    "f1": f1,
                                    "em": EM,
                                    "chrf": chrf,
                                    "bleu": bleu
                                })

                            # Save per-row result
                            row_data = {
                                "id": pred_data.get("id", "unknown"),
                                "en": pred_data.get("en", "unknown"),
                                "scores": row_scores
                            }
                            results_list.append(row_data)

                        except json.JSONDecodeError as e:
                            print(f"Skipping a corrupted line: {e}")
                            continue

            except FileNotFoundError as e:
                print(f"File not found during open: {e}")

            # === OUTPUT: Include la cartella della pipeline ===
            jsonl_output_file = os.path.join(results_dir, "evaluation_FIXED", "string-comparison", pipeline, f"en-{language}", f"{perturbation}.jsonl")
            
            os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)
            with open(jsonl_output_file, "w", encoding="utf-8") as jsonl_file:
             for row in results_list:
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")