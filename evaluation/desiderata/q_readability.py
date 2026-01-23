import json
import os
import textstat
import numpy as np


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
        print("File: ", jsonl_file)

        total_entries = 0
        readability_scores = []

        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
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
     
                if len(questions) == 0:
                    continue

                total_entries += 1
                instance_scores = []

                # Ensure all questions are strings
                questions = [str(q) if not isinstance(q, str) else q for q in questions]

                for question in questions:
                    score = textstat.flesch_reading_ease(question)  # Flesch Reading Ease Score
                    instance_scores.append(score)

                avg_instance_score = np.mean(instance_scores)
                readability_scores.append(avg_instance_score)


        def classify_readability(score):
            if score >= 90:
                return "Very Easy (5th grade)"
            elif score >= 80:
                return "Easy (6th grade)"
            elif score >= 70:
                return "Fairly Easy (7th grade)"
            elif score >= 60:
                return "Standard (8th-9th grade)"
            elif score >= 50:
                return "Fairly Difficult (10th-12th grade)"
            elif score >= 30:
                return "Difficult (College)"
            else:
                return "Very Difficult (Graduate level)"

        if readability_scores:
            avg_readability = np.mean(readability_scores)
            print(f"Average Readability Score (Flesch-Kincaid): {avg_readability:.2f}")
        else:
            print("No valid questions found in dataset.")
        print("Division: ", classify_readability(avg_readability))
