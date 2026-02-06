"""
Reasoning-Augmented QA for ASKQE Extension

This script runs Question Answering with a reasoning-augmented prompt.
Results are saved in the extension's results folder.

Usage:
    python qwen-3b-reasoning.py --run_all
"""

import torch
import json
import os
import argparse
from prompt_reasoning import qa_prompt_reasoning
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

# All configurations
PIPELINES = ["atomic", "semantic", "vanilla"]
LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PERTURBATIONS = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]


def load_backtranslations(project_root, lang="es", perturbation="alteration"):
    """Load backtranslations from google_translate files and return a dict keyed by id."""
    bt_data = {}

    bt_file = os.path.join(project_root, "backtranslation", f"en-{lang}", f"bt-{perturbation}.jsonl")
    
    if os.path.exists(bt_file):
        with open(bt_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                bt_data[data['id']] = {
                    'source': data.get('en', ''),
                    'bt': data.get(f'bt_pert_{lang}', ''),
                    'target': data.get(f'pert_{lang}', '')
                }
    else:
        print(f"Warning: Backtranslation file not found: {bt_file}")
    
    return bt_data


def process_qa(tokenizer, model, device, qg_file, output_file, bt_data, sentence_type):
    """Process QA for a single configuration using reasoning prompt."""
    
    if os.path.exists(output_file):
        print(f"Output file already exists, skipping: {output_file}")
        return
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(qg_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            sentence = None
            data_id = data.get('id', '')
            
            if sentence_type in ['bt', 'source'] and data_id in bt_data:
                sentence = bt_data[data_id].get(sentence_type, None)
            else:
                sentence = data.get(sentence_type, None)
            
            questions = data.get("questions", None)

            if sentence and questions:
                # Use reasoning-augmented prompt
                prompt = qa_prompt_reasoning.replace("{{sentence}}", sentence).replace("{{questions}}", questions)

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
                
                print(f"> {generated_answers[:100]}...")

                data['answers'] = generated_answers
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all", action="store_true", help="Run all configurations automatically")
    parser.add_argument("--lang", type=str, default="es", help="Language: es, fr, hi, tl, zh")
    parser.add_argument("--perturbation", type=str, default="alteration", help="Perturbation type")
    parser.add_argument("--pipeline", type=str, default="vanilla", help="Pipeline: vanilla, atomic, semantic")

    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(extension_dir)
    
    # QG input comes from baseline results
    qg_results_dir = os.path.join(project_root, "results Qwen3B baseline", "QG")
    
    # Output goes to extension results folder
    results_dir = os.path.join(extension_dir, "results")

    print(f"Extension dir: {extension_dir}")
    print(f"QG input from: {qg_results_dir}")
    print(f"Results output to: {results_dir}")
    print("=" * 60)
    print("REASONING-AUGMENTED QA")
    print("=" * 60)

    if args.run_all:
        # Run all configurations automatically
        total_configs = len(PIPELINES) * len(LANGUAGES) * len(PERTURBATIONS) * 2  # 2 for source/bt
        current = 0
        
        for pipeline in PIPELINES:
            qg_file = os.path.join(qg_results_dir, f"{pipeline}_qwen-3b.jsonl")
            
            if not os.path.exists(qg_file):
                print(f"Skipping {pipeline}: QG file not found at {qg_file}")
                continue
            
            # First, run source-based QA (only once per pipeline)
            print(f"\n{'='*60}")
            print(f"Processing SOURCE for pipeline: {pipeline}")
            print(f"{'='*60}")
            
            output_file = os.path.join(results_dir, "QA", "source", f"en-{pipeline}.jsonl")
            bt_data = {}  # No backtranslation needed for source
            
            process_qa(tokenizer, model, device, qg_file, output_file, bt_data, "en")
            print(f"Saved: {output_file}")
            
            # Then, run BT-based QA for all languages and perturbations
            for lang in LANGUAGES:
                for perturbation in PERTURBATIONS:
                    current += 1
                    print(f"\n{'='*60}")
                    print(f"[{current}/{total_configs}] Processing BT: {pipeline} / {lang} / {perturbation}")
                    print(f"{'='*60}")
                    
                    bt_data = load_backtranslations(project_root, lang, perturbation)
                    
                    if not bt_data:
                        print(f"Skipping: No backtranslation data")
                        continue
                    
                    output_file = os.path.join(results_dir, "QA", "bt", f"{lang}-{pipeline}-{perturbation}.jsonl")
                    
                    process_qa(tokenizer, model, device, qg_file, output_file, bt_data, "bt")
                    print(f"Saved: {output_file}")
        
        print(f"\n{'='*60}")
        print("All configurations completed!")
        print(f"{'='*60}")
    
    else:
        # Single run mode
        qg_file = os.path.join(qg_results_dir, f"{args.pipeline}_qwen-3b.jsonl")
        
        bt_data = load_backtranslations(project_root, args.lang, args.perturbation)
        print(f"Loaded {len(bt_data)} backtranslation entries")
        
        output_file = os.path.join(results_dir, "QA", "bt", f"{args.lang}-{args.pipeline}-{args.perturbation}.jsonl")
        process_qa(tokenizer, model, device, qg_file, output_file, bt_data, "bt")


if __name__ == "__main__":
    main()
