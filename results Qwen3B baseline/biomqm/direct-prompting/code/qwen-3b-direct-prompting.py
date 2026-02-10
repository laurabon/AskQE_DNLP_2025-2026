"""
BIOMQM Question Answering Script - Direct Prompting (Reverse CoT)
Generates answers using XML-structured direct prompts.
Each question is answered individually (one API call per question).

Usage:
  # Generate source answers
  python qwen-3b-direct-prompting.py --mode source --qg_input_path /path/to/qg.jsonl --output_path /path/to/output.jsonl

  # Generate bt answers per language
  python qwen-3b-direct-prompting.py --mode bt --lang de --qg_input_path /path/to/qg.jsonl --output_path /path/to/output.jsonl
"""

import torch
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]

# === PROMPT DEFINITIONS (XML-structured) ===

# Source prompt: direct prompting without error warning
source_prompt = """Based on the context, answer the question.
<context>{sentence}</context>
<question>{question}</question>"""

# Backtranslation prompt: includes warning about potential translation errors
bt_prompt = """Based on the context, answer the question.
Note: The context may contain translation errors.
<context>{sentence}</context>
<question>{question}</question>"""


def parse_questions(questions_str):
    """
    Parse questions from string to list.
    Questions can be stored as JSON string or already as list.
    """
    if isinstance(questions_str, list):
        return questions_str
    
    if not questions_str or questions_str.strip() == "":
        return []
    
    try:
        questions = json.loads(questions_str)
        if isinstance(questions, list):
            return questions
        return [str(questions)]
    except json.JSONDecodeError:
        # If not valid JSON, treat as single question
        return [questions_str.strip()]


def generate_single_answer(tokenizer, model, device, sentence, question, is_bt=False):
    """
    Generate answer for a single sentence and question pair.
    Uses different prompts for source vs backtranslation.
    
    Args:
        tokenizer: Tokenizer instance
        model: Model instance
        device: Device to use
        sentence: Context sentence
        question: Single question string
        is_bt: If True, use bt_prompt with error warning
    
    Returns:
        Generated answer string
    """
    # Select appropriate prompt
    prompt_template = bt_prompt if is_bt else source_prompt
    prompt = prompt_template.format(sentence=sentence, question=question)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions directly and concisely based on the given context."},
        {"role": "user", "content": prompt},
    ]
    
    # Use 2-step tokenization for robustness: template -> string -> tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted string
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Unpack model_inputs (contains input_ids and attention_mask)
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    
    response = outputs[0][model_inputs.input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    
    if answer:
        answer = answer.strip().strip('"\'')
    
    return answer


def generate_answers_for_questions(tokenizer, model, device, sentence, questions_str, is_bt=False):
    """
    Generate answers for all questions related to a sentence.
    Makes one API call per question.
    
    Args:
        tokenizer: Tokenizer instance
        model: Model instance
        device: Device to use
        sentence: Context sentence
        questions_str: Questions as JSON string or list
        is_bt: If True, use bt_prompt
    
    Returns:
        List of answers (one per question)
    """
    questions = parse_questions(questions_str)
    
    if not questions:
        return []
    
    answers = []
    for q in questions:
        answer = generate_single_answer(tokenizer, model, device, sentence, q, is_bt=is_bt)
        answers.append(answer)
    
    return answers


def process_source_qa(tokenizer, model, device, qg_file, output_file):
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
    
    # Second pass: generate answers for unprocessed unique src
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (src, data) in enumerate(unique_src.items()):
            if src in processed_src:
                continue
            
            questions = parse_questions(data['questions'])
            print(f"[{i+1}/{len(unique_src)}] Processing src with {len(questions)} questions...")
            
            # Generate answers (one call per question, is_bt=False for source)
            answers = generate_answers_for_questions(
                tokenizer, model, device,
                data['src'], data['questions'],
                is_bt=False
            )
            
            if answers:
                print(f"> First answer: {answers[0][:60]}...")
            
            output_row = {
                'src': data['src'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes']
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\nSource QA completed. Output: {output_file}")


def process_bt_qa(tokenizer, model, device, qg_file, output_file, lang):
    """
    Process BT QA for a specific language.
    Generate answers for unique (src, bt_tgt) pairs only.
    Uses bt_prompt with error warning.
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
    
    # Second pass: generate answers for unprocessed unique bt
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (key, data) in enumerate(unique_bt.items()):
            if key in processed_keys:
                continue
            
            if limit and i >= limit:
                print(f"Limit of {limit} reached. Stopping.")
                break
            
            questions = parse_questions(data['questions'])
            print(f"[{i+1}/{len(unique_bt)}] Processing bt for {lang} with {len(questions)} questions...")
            
            # Generate answers (one call per question, is_bt=True for backtranslation)
            answers = generate_answers_for_questions(
                tokenizer, model, device,
                data['bt_tgt'], data['questions'],
                is_bt=True  # Use bt_prompt with error warning
            )
            
            if answers:
                print(f"> First answer: {answers[0][:60]}...")
            
            output_row = {
                'src': data['src'],
                'bt_tgt': data['bt_tgt'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes']
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\nBT QA for {lang} completed. Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="BIOMQM QA Script - Direct Prompting")
    parser.add_argument("--mode", type=str, required=True, choices=["source", "bt"],
                        help="Mode: 'source' for source QA, 'bt' for backtranslation QA")
    parser.add_argument("--lang", type=str, choices=LANGUAGES,
                        help="Language for bt mode (required when mode=bt)")
    parser.add_argument("--pipeline", type=str, default="direct-prompting",
                        help="Pipeline name (default: direct-prompting)")
    parser.add_argument("--qg_input_path", type=str, required=True,
                        help="Path to QG input file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of items to process (for testing)")
    args = parser.parse_args()
    
    if args.mode == "bt" and not args.lang:
        parser.error("--lang is required when --mode is 'bt'")
    
    # Validate input file exists
    if not os.path.exists(args.qg_input_path):
        print(f"QG file not found: {args.qg_input_path}")
        return
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Pipeline: {args.pipeline}")
    if args.limit:
        print(f"TEST MODE: limiting to {args.limit} items")
    print(f"Generation params: temperature=0.1, top_p=0.9, repetition_penalty=1.1")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Run appropriate mode
    if args.mode == "source":
        print("\n=== SOURCE QA (Direct Prompting) ===")
        print("Using source prompt (no error warning)")
        process_source_qa(tokenizer, model, device, args.qg_input_path, args.output_path, limit=args.limit)
    
    elif args.mode == "bt":
        print(f"\n=== BT QA for {args.lang} (Direct Prompting) ===")
        print("Using bt prompt (with error warning)")
        process_bt_qa(tokenizer, model, device, args.qg_input_path, args.output_path, args.lang, limit=args.limit)


if __name__ == "__main__":
    main()
