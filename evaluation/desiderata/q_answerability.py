import torch
import json
import os
import numpy as np
from transformers import LongformerTokenizer, LongformerForSequenceClassification


model_name = "potsawee/longformer-large-4096-answerable-squad2"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerForSequenceClassification.from_pretrained(model_name)


#pipelines = ["atomic", "semantic", "vanilla"]
models = ["qwen-3b"]
pipelines = ["atomic"]

# Use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
results_dir = os.path.join(project_root, "results Qwen3B baseline")

for pipeline in pipelines:
    for model_name in models:
        jsonl_file = os.path.join(results_dir, "QG", f"{pipeline}_{model_name}.jsonl")
        output_file = os.path.join(script_dir, "answerability", f"{pipeline}_{model_name}.jsonl")

        print(f"\nProcessing File: {jsonl_file}")

        answerability_scores = []
        total_questions = 0
        processed_data = []

        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    context = data.get("en", "")
                    questions = data.get("questions", [])

                    if isinstance(questions, str):
                        try:
                            questions = json.loads(questions)
                            if not isinstance(questions, list):
                                print(f"Skipping due to invalid question format.")
                                continue 
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"Skipping due to invalid question format: {e}")
                            continue

                    if not context or not questions:
                        continue

                    instance_scores = []
                    question_scores = []

                    # Ensure all questions are strings
                    questions = [str(q) if not isinstance(q, str) else q for q in questions]

                    for question in questions:
                        input_text = question + ' ' + tokenizer.sep_token + ' ' + context
                        inputs = tokenizer(input_text, max_length=4096, truncation=True, return_tensors="pt")

                        prob = torch.sigmoid(model(**inputs).logits.squeeze(-1))
                        answerability = prob.item() * 100
                        instance_scores.append(answerability)
                        total_questions += 1
                        question_scores.append({"question": question, "answerability_score": answerability})

                    if instance_scores:
                        avg_instance_score = np.mean(instance_scores)
                        answerability_scores.append(avg_instance_score)
                    print(question_scores)
                    print(avg_instance_score)

                    data["answerability_scores"] = avg_instance_score
                    data["answerability_avg"] = avg_instance_score
                    processed_data.append(data)

                except json.JSONDecodeError as e:
                    print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                    continue
        output_file = os.path.join(results_dir, "evaluation", "desiderata", "answerability", f"{pipeline}_{model_name}.jsonl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as out_f:
            for entry in processed_data:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if answerability_scores:
            avg_answerability = np.mean(answerability_scores)
            print(f"\nAverage Answerability Score: {avg_answerability:.2f}%")
        else:
            print("\nNo valid questions found in dataset.")
