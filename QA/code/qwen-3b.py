import torch
import json
import os
import argparse
from prompt import qa_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

def load_backtranslations(lang="es"):
    """Load backtranslations from google_translate files and return a dict keyed by id."""
    bt_data = {}

    # Use relative path from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    bt_file = os.path.join(project_root, "backtranslation", f"en-{lang}", "bt-alteration.jsonl")
    
    if os.path.exists(bt_file):
        with open(bt_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # bt_pert_es is the backtranslation of the perturbed Spanish
                bt_data[data['id']] = {
                    'source': data.get('en', ''),           # Original English
                    'bt': data.get(f'bt_pert_{lang}', ''),  # Backtranslation of perturbed translation
                    'target': data.get(f'pert_{lang}', '')  # Perturbed translation
                }
    else:
        print(f"Warning: Backtranslation file not found: {bt_file}")
    
    return bt_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--sentence_type", type=str, help="Type of sentence to use: 'en', 'source', 'bt' (backtranslation)")
    parser.add_argument("--lang", type=str, default="es", help="Language for backtranslation: es, fr, hi, tl, zh")

    args = parser.parse_args()

    # Load backtranslations if needed
    bt_data = {}
    if args.sentence_type in ['bt', 'source']:
        print(f"Loading backtranslations for language: {args.lang}")
        bt_data = load_backtranslations(args.lang)
        print(f"Loaded {len(bt_data)} backtranslation entries")

    processed_sentences = set()

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as output_file:
            for line in output_file:
                data = json.loads(line.strip())
                processed_sentences.add(data["id"])

    # =========================================== Load Dataset ===========================================
    pipeline_types = ["vanilla"]

    # Use relative path from QA/code directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    for pipeline_type in pipeline_types:
        qg_file = os.path.join(project_root, "QG", "qwen-3b", f"{pipeline_type}_qwen-3b.jsonl")
        output_file = f"{args.output_path}-{pipeline_type}.jsonl"
        
        with open(qg_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line)
                
                # Get sentence based on sentence_type
                sentence = None
                data_id = data.get('id', '')
                
                if args.sentence_type in ['bt', 'source'] and data_id in bt_data:
                    # Use backtranslation data
                    sentence = bt_data[data_id].get(args.sentence_type, None)
                else:
                    # Use field directly from QG data (e.g., 'en')
                    sentence = data.get(args.sentence_type, None)
                
                questions = data.get("questions", None)

                if sentence and questions:
                    prompt_template = qa_prompt
                    prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{questions}}", questions)

                    print(prompt)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    input_ids = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=1024,
                        )
                    response = outputs[0][input_ids.shape[-1]:]
                    generated_answers = tokenizer.decode(response, skip_special_tokens=True)

                    if generated_answers:
                        generated_answers = generated_answers.strip('"\'')
                    
                    print(f"> {generated_answers}")
                    print("\n======================================================\n")

                    data['answers'] = generated_answers
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                else:
                    pass


if __name__ == "__main__":
    main()
