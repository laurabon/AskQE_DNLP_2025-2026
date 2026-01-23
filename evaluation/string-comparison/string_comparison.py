import json
import nltk
import os
from utils import compare_answers

nltk.download("punkt")

'''languages = ["es", "fr", "hi", "tl", "zh"]
pipelines = ["atomic", "semantic", "vanilla"]'''


languages = ["es", "fr"]
pipelines = ["atomic"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact",
                 "intensifier", "expansion_impact", "omission", "alteration"]

# Use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(project_root, "results Qwen3B baseline")

for language in languages:
    for pipeline in pipelines:
        for perturbation in perturbations:
            predicted_file = os.path.join(results_dir, "QA", "bt", f"{language}-{pipeline}-{perturbation}.jsonl")
            reference_file = os.path.join(results_dir, "QA", "source", f"en-{pipeline}.jsonl")

            results_list = []
            try:
                with open(predicted_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file:
                    for pred_line, ref_line in zip(pred_file, ref_file):
                        try:
                            pred_data = json.loads(pred_line)
                            ref_data = json.loads(ref_line)

                            predicted_answers = pred_data.get("answers", [])
                            reference_answers = ref_data.get("answers", [])

                            if isinstance(predicted_answers, str):
                                try:
                                    predicted_answers = json.loads(predicted_answers)
                                except json.JSONDecodeError:
                                    continue

                            if isinstance(reference_answers, str):
                                try:
                                    reference_answers = json.loads(reference_answers)
                                except json.JSONDecodeError:
                                    continue

                            if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                                continue
                            if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                                continue

                            row_scores = []
                            for pred, ref in zip(predicted_answers, reference_answers):
                                # Convert non-string values to strings
                                if not isinstance(pred, str):
                                    pred = str(pred) if pred is not None else ""
                                if not isinstance(ref, str):
                                    ref = str(ref) if ref is not None else ""
                                if not pred.strip() or not ref.strip():
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
                            print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                            continue

            except FileNotFoundError as e:
                print(f"File not found: {e}")

            jsonl_output_file = os.path.join(results_dir, "evaluation", "string-comparison", f"en-{language}", f"{perturbation}.jsonl")
            os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)
            with open(jsonl_output_file, "w", encoding="utf-8") as jsonl_file:
             for row in results_list:
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")