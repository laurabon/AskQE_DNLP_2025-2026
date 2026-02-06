"""
Reasoning-Augmented QA for BioMQM Dataset (Optimized)

Uses deduplication to avoid redundant inference:
- Source mode: processes unique src values only
- BT mode: processes unique (src, bt_tgt) pairs only

Includes resume capability and row_indexes tracking.

Usage:
  # Generate source answers (unique src only)
  python biomqm_reasoning.py --mode source --pipeline vanilla

  # Generate bt answers per language (unique per src+bt_tgt)
  python biomqm_reasoning.py --mode bt --lang de --pipeline vanilla
  
  # Quick test with limited samples
  python biomqm_reasoning.py --mode source --pipeline vanilla --max_samples 10
"""

import torch
import json
import os
import argparse
from prompt_reasoning import qa_prompt_reasoning
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]


def generate_answer(tokenizer, model, device, sentence, questions):
    """Generate answers using reasoning-augmented prompt."""
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
    
    return generated_answers


def process_source_qa(tokenizer, model, device, qg_file, output_file, max_samples=None):
    """
    Process source QA: generate answers for unique src values only.
    Output includes row_indexes for later mapping.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # First pass: collect unique src and their row indexes
    print("Collecting unique source sentences...")
    unique_src = {}  # key: src -> {data, row_indexes}
    
    with open(qg_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            src = data.get('src', '')
            
            if src not in unique_src:
                unique_src[src] = {
                    'src': src,
                    'lang_tgt': data.get('lang_tgt', ''),
                    'questions': data.get('questions', ''),
                    'row_indexes': [idx]
                }
            else:
                unique_src[src]['row_indexes'].append(idx)
    
    print(f"Found {len(unique_src)} unique src values from {idx + 1} total rows")
    
    # Check for resume
    processed_src = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_src.add(data.get('src', ''))
        print(f"Resuming: {len(processed_src)} already processed")
    
    # Apply max_samples limit
    items_to_process = [(src, data) for src, data in unique_src.items() if src not in processed_src]
    if max_samples:
        items_to_process = items_to_process[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    # Second pass: generate answers for unprocessed unique src
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (src, data) in enumerate(items_to_process):
            print(f"[{i+1}/{len(items_to_process)}] Processing src...")
            print(f"  Source: {data['src'][:80]}...")
            
            answers = generate_answer(
                tokenizer, model, device,
                data['src'], data['questions']
            )
            
            print(f"  > {answers[:80]}...")
            
            output_row = {
                'src': data['src'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes']
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Source QA completed. Output: {output_file}")


def process_bt_qa(tokenizer, model, device, qg_file, output_file, lang, max_samples=None):
    """
    Process BT QA for a specific language.
    Generate answers for unique (src, bt_tgt) pairs only.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # First pass: collect unique (src, bt_tgt) pairs and their row indexes
    print(f"Collecting unique bt_tgt for language: {lang}...")
    unique_bt = {}  # key: (src, bt_tgt) -> {data, row_indexes}
    
    with open(qg_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            if data.get('lang_tgt', '') != lang:
                continue
            
            src = data.get('src', '')
            bt_tgt = data.get('bt_tgt', '')
            key = (src, bt_tgt)
            
            if key not in unique_bt:
                unique_bt[key] = {
                    'src': src,
                    'bt_tgt': bt_tgt,
                    'lang_tgt': lang,
                    'questions': data.get('questions', ''),
                    'row_indexes': [idx]
                }
            else:
                unique_bt[key]['row_indexes'].append(idx)
    
    total_rows = sum(len(d['row_indexes']) for d in unique_bt.values())
    print(f"Found {len(unique_bt)} unique (src, bt_tgt) pairs from {total_rows} rows for {lang}")
    
    # Check for resume
    processed_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = (data.get('src', ''), data.get('bt_tgt', ''))
                processed_keys.add(key)
        print(f"Resuming: {len(processed_keys)} already processed")
    
    # Apply max_samples limit
    items_to_process = [(key, data) for key, data in unique_bt.items() if key not in processed_keys]
    if max_samples:
        items_to_process = items_to_process[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    # Second pass: generate answers for unprocessed unique bt
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (key, data) in enumerate(items_to_process):
            print(f"[{i+1}/{len(items_to_process)}] Processing bt for {lang}...")
            print(f"  BT: {data['bt_tgt'][:80]}...")
            
            answers = generate_answer(
                tokenizer, model, device,
                data['bt_tgt'], data['questions']
            )
            
            print(f"  > {answers[:80]}...")
            
            output_row = {
                'src': data['src'],
                'bt_tgt': data['bt_tgt'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes']
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\n✅ BT QA for {lang} completed. Output: {output_file}")


def main():
    print("=" * 70)
    print("REASONING-AUGMENTED QA FOR BIOMQM (OPTIMIZED)")
    print("=" * 70)
    
    parser = argparse.ArgumentParser(description="BioMQM Reasoning QA Script")
    parser.add_argument("--mode", type=str, required=True, choices=["source", "bt"],
                        help="Mode: 'source' for source QA, 'bt' for backtranslation QA")
    parser.add_argument("--lang", type=str, choices=LANGUAGES,
                        help="Language for bt mode (required when mode=bt)")
    parser.add_argument("--pipeline", type=str, default="vanilla",
                        help="Pipeline name (default: vanilla)")
    parser.add_argument("--qg_input_path", type=str,
                        help="Custom path to QG input file")
    parser.add_argument("--output_path", type=str,
                        help="Custom output path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()
    
    if args.mode == "bt" and not args.lang:
        parser.error("--lang is required when --mode is 'bt'")
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(extension_dir)
    
    # Results go to extension folder
    results_dir = os.path.join(extension_dir, "results", "biomqm")
    
    # QG input file (from baseline results)
    if args.qg_input_path:
        qg_file = args.qg_input_path
    else:
        qg_file = os.path.join(project_root, "results Qwen3B baseline", "QG", "biomqm", f"{args.pipeline}_qwen-3b.jsonl")
    
    if not os.path.exists(qg_file):
        print(f"❌ QG file not found: {qg_file}")
        print("Run Question Generation for BioMQM first.")
        return
    
    print(f"QG input: {qg_file}")
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("\n⚠️  WARNING: Running on CPU will be very slow!")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    
    # Run appropriate mode
    if args.mode == "source":
        if args.output_path:
            output_file = args.output_path
        else:
            output_file = os.path.join(results_dir, "unique", f"source-{args.pipeline}.jsonl")
        
        process_source_qa(tokenizer, model, device, qg_file, output_file, args.max_samples)
    
    elif args.mode == "bt":
        if args.output_path:
            output_file = args.output_path
        else:
            output_file = os.path.join(results_dir, "unique", f"bt-{args.lang}-{args.pipeline}.jsonl")
        
        process_bt_qa(tokenizer, model, device, qg_file, output_file, args.lang, args.max_samples)


if __name__ == "__main__":
    main()
