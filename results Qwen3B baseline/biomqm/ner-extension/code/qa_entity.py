"""
Entity-Aware Question Answering for BioMQM

Answers entity-specific questions on source and backtranslated sentences.

Usage:
    python qa_entity.py --mode source --input_path /path/to/qg.jsonl --output_path /path/to/qa_source.jsonl
    python qa_entity.py --mode bt --lang de --input_path /path/to/qg.jsonl --bt_path /path/to/bt.jsonl --output_path /path/to/qa_bt.jsonl
"""

import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]

# Prompt for answering entity-specific questions
QA_PROMPT = """Task: Answer the question based ONLY on the given sentence.

Sentence: {sentence}
Question: {question}

Instructions:
- Answer using ONLY information from the sentence
- If the answer is not in the sentence, respond with "[NOT FOUND]"
- Be concise and direct

Answer:"""


def generate_answer(tokenizer, model, device, sentence, question):
    """Generate an answer for a question about the sentence."""
    prompt = QA_PROMPT.format(sentence=sentence, question=question)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based only on given context."},
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
            max_new_tokens=64,
        )
    
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True).strip()
    
    return answer


def process_source_qa(tokenizer, model, device, qg_data, output_path, max_samples=None):
    """Process QA for source sentences (unique only)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if max_samples:
        qg_data = qg_data[:max_samples]
    
    print(f"Processing {len(qg_data)} samples for source QA...")
    
    processed = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for data in qg_data:
            processed += 1
            
            if processed % 20 == 0:
                print(f"[{processed}/{len(qg_data)}] Processing source...")
            
            src = data.get('src', '')
            questions = data.get('questions', [])
            row_indexes = data.get('row_indexes', [])
            
            answers = []
            for q_info in questions:
                question = q_info.get('question', '')
                entity_type = q_info.get('entity_type', 'UNKNOWN')
                entity_text = q_info.get('entity_text', '')
                
                answer = generate_answer(tokenizer, model, device, src, question)
                
                answers.append({
                    'question': question,
                    'entity_type': entity_type,
                    'entity_text': entity_text,
                    'answer': answer
                })
            
            output_row = {
                'src': src,
                'answers': answers,
                'row_indexes': row_indexes
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"Source QA complete. Output: {output_path}")


def process_bt_qa(tokenizer, model, device, qg_data, bt_lookup, output_path, lang, max_samples=None):
    """Process QA for backtranslated sentences."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter QG data for this language and build unique (src, bt) pairs
    unique_bt = {}  # key: (src, bt_tgt) -> {data, row_indexes}
    
    for data in qg_data:
        src = data.get('src', '')
        row_indexes = data.get('row_indexes', [])
        
        for row_idx in row_indexes:
            bt_info = bt_lookup.get(row_idx, {})
            if bt_info.get('lang_tgt') != lang:
                continue
            
            bt_tgt = bt_info.get('bt_tgt', '')
            key = (src, bt_tgt)
            
            if key not in unique_bt:
                unique_bt[key] = {
                    'src': src,
                    'bt_tgt': bt_tgt,
                    'questions': data.get('questions', []),
                    'row_indexes': [row_idx]
                }
            else:
                unique_bt[key]['row_indexes'].append(row_idx)
    
    bt_items = list(unique_bt.values())
    if max_samples:
        bt_items = bt_items[:max_samples]
    
    print(f"Processing {len(bt_items)} unique BT samples for {lang}...")
    
    processed = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for data in bt_items:
            processed += 1
            
            if processed % 20 == 0:
                print(f"[{processed}/{len(bt_items)}] Processing BT {lang}...")
            
            bt_tgt = data.get('bt_tgt', '')
            questions = data.get('questions', [])
            row_indexes = data.get('row_indexes', [])
            
            answers = []
            for q_info in questions:
                question = q_info.get('question', '')
                entity_type = q_info.get('entity_type', 'UNKNOWN')
                entity_text = q_info.get('entity_text', '')
                
                answer = generate_answer(tokenizer, model, device, bt_tgt, question)
                
                answers.append({
                    'question': question,
                    'entity_type': entity_type,
                    'entity_text': entity_text,
                    'answer': answer
                })
            
            output_row = {
                'src': data.get('src', ''),
                'bt_tgt': bt_tgt,
                'answers': answers,
                'row_indexes': row_indexes
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"BT QA for {lang} complete. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Entity-Aware QA for BioMQM")
    parser.add_argument("--mode", type=str, required=True, choices=["source", "bt"],
                        help="Mode: 'source' or 'bt'")
    parser.add_argument("--lang", type=str, choices=LANGUAGES,
                        help="Language for BT mode")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to QG output JSONL file")
    parser.add_argument("--bt_path", type=str,
                        help="Path to original backtranslation file (for BT mode)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()
    
    if args.mode == "bt" and not args.lang:
        parser.error("--lang is required for BT mode")
    
    if args.mode == "bt" and not args.bt_path:
        parser.error("--bt_path is required for BT mode")
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    
    # Load QG data
    qg_data = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                qg_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(qg_data)} QG samples")
    
    if args.mode == "source":
        process_source_qa(tokenizer, model, device, qg_data, args.output_path, args.max_samples)
    
    elif args.mode == "bt":
        # Load BT lookup
        bt_lookup = {}  # row_idx -> {bt_tgt, lang_tgt}
        with open(args.bt_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    bt_lookup[idx] = {
                        'bt_tgt': data.get('bt_tgt', ''),
                        'lang_tgt': data.get('lang_tgt', '')
                    }
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(bt_lookup)} BT entries")
        
        process_bt_qa(tokenizer, model, device, qg_data, bt_lookup, args.output_path, args.lang, args.max_samples)


if __name__ == "__main__":
    main()
