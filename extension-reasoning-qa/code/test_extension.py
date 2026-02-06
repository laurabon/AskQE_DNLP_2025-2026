"""
Quick Test Script for Reasoning-Augmented QA Extension

Tests the extension with:
- 1 language (es)
- 1 pipeline (atomic)
- 1 perturbation (alteration)
- Only 5 samples

Usage:
    python test_extension.py
"""

import torch
import json
import os
from prompt_reasoning import qa_prompt_reasoning
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"
MAX_SAMPLES = 5  # Only process 5 samples for testing


def load_backtranslations(project_root, lang="es", perturbation="alteration"):
    """Load backtranslations."""
    bt_data = {}
    bt_file = os.path.join(project_root, "backtranslation", f"en-{lang}", f"bt-{perturbation}.jsonl")
    
    if os.path.exists(bt_file):
        with open(bt_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                bt_data[data['id']] = {
                    'source': data.get('en', ''),
                    'bt': data.get(f'bt_pert_{lang}', ''),
                }
    return bt_data


def process_qa_sample(tokenizer, model, device, sentence, questions):
    """Process a single QA sample."""
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
            max_new_tokens=512,
        )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip('"\'')


def main():
    print("=" * 60)
    print("REASONING QA EXTENSION - QUICK TEST")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("\n⚠️  WARNING: Running on CPU will be very slow!")
        print("    Consider using a GPU for actual experiments.\n")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(extension_dir)
    
    # Test configuration
    lang = "es"
    pipeline = "atomic"
    perturbation = "alteration"
    
    qg_file = os.path.join(project_root, "results Qwen3B baseline", "QG", f"{pipeline}_qwen-3b.jsonl")
    
    if not os.path.exists(qg_file):
        print(f"ERROR: QG file not found: {qg_file}")
        return
    
    bt_data = load_backtranslations(project_root, lang, perturbation)
    print(f"Loaded {len(bt_data)} backtranslation entries")
    
    # Output files
    output_dir = os.path.join(extension_dir, "results", "test")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"test_{lang}_{pipeline}_{perturbation}.jsonl")
    
    print(f"\nTest config: {lang} / {pipeline} / {perturbation}")
    print(f"Processing {MAX_SAMPLES} samples...")
    print("-" * 60)
    
    results = []
    count = 0
    
    with open(qg_file, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= MAX_SAMPLES:
                break
            
            data = json.loads(line)
            data_id = data.get('id', '')
            questions = data.get("questions", None)
            
            # Get source sentence
            source_sentence = data.get('en', '')
            
            # Get BT sentence
            bt_sentence = bt_data.get(data_id, {}).get('bt', '')
            
            if not source_sentence or not questions or not bt_sentence:
                continue
            
            count += 1
            print(f"\n[{count}/{MAX_SAMPLES}] Processing ID: {data_id}")
            print(f"Source: {source_sentence[:80]}...")
            print(f"BT:     {bt_sentence[:80]}...")
            
            # Get answers from source
            print("  → Source answers...")
            source_answers = process_qa_sample(tokenizer, model, device, source_sentence, questions)
            
            # Get answers from BT
            print("  → BT answers...")
            bt_answers = process_qa_sample(tokenizer, model, device, bt_sentence, questions)
            
            print(f"  Source: {source_answers[:100]}...")
            print(f"  BT:     {bt_answers[:100]}...")
            
            results.append({
                "id": data_id,
                "source_sentence": source_sentence,
                "bt_sentence": bt_sentence,
                "questions": questions,
                "source_answers": source_answers,
                "bt_answers": bt_answers
            })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 60)
    print(f"✅ Test completed! Processed {len(results)} samples")
    print(f"Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
