import json
import os


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
        duplicate_questions_count = 0

        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                total_entries += 1
                questions = data.get("questions", [])

                if isinstance(questions, str):
                    try:
                        questions = json.loads(questions)
                        if not isinstance(questions, list):
                            continue
                    except (json.JSONDecodeError, ValueError) as e:
                        continue

                # Convert elements to hashable types (tuple for lists, or str) to avoid TypeError
                hashable_questions = []
                for q in questions:
                    if isinstance(q, list):
                        hashable_questions.append(tuple(q))
                    else:
                        hashable_questions.append(q)
                
                unique_questions = set(hashable_questions)
                if len(unique_questions) < len(questions):
                    duplicate_questions_count += 1

        print(f"Duplicate Questions: {duplicate_questions_count} / {total_entries}")
