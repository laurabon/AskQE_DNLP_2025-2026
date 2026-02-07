"""
Entity-Aware Question Generation for BioMQM

Uses Qwen to generate questions specific to each extracted entity.

Usage:
    python qg_entity_aware.py --input_path /path/to/ner_output.jsonl --output_path /path/to/qg_output.jsonl
"""

import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

# Prompt for entity-aware question generation
QG_ENTITY_PROMPT = """Task: Generate a specific question about the given entity based on the sentence.

The entity is: "{entity}" (Type: {entity_type})
The sentence is: "{sentence}"

Generate exactly ONE clear, specific question that:
1. Can be answered using ONLY information from the sentence
2. Specifically asks about the entity mentioned
3. The answer should be the entity itself or information directly related to it

Output only the question, nothing else.

Question:"""


def generate_question(tokenizer, model, device, sentence, entity_text, entity_type):
    """Generate a question for a specific entity."""
    prompt = QG_ENTITY_PROMPT.format(
        entity=entity_text,
        entity_type=entity_type,
        sentence=sentence
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates clear, answerable questions."},
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
            temperature=0.3,
            do_sample=True,
        )
    
    response = outputs[0][input_ids.shape[-1]:]
    question = tokenizer.decode(response, skip_special_tokens=True).strip()
    
    # Clean up the question
    question = question.replace('"', '').replace("'", "")
    if not question.endswith('?'):
        question += '?'
    
    return question


def main():
    parser = argparse.ArgumentParser(description="Entity-Aware QG for BioMQM")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to NER output JSONL file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file with questions")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()
    
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
    
    # Process file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"\nProcessing: {args.input_path}")
    
    # Load input data
    data_list = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if args.max_samples:
        data_list = data_list[:args.max_samples]
    
    print(f"Loaded {len(data_list)} samples")
    
    # Generate questions
    processed = 0
    total_questions = 0
    
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for data in data_list:
            processed += 1
            
            src = data.get('src', '')
            entities = data.get('entities', [])
            row_indexes = data.get('row_indexes', [])
            
            if processed % 20 == 0:
                print(f"[{processed}/{len(data_list)}] Processing... ({total_questions} questions generated)")
            
            questions = []
            
            # Generate one question per entity
            for entity in entities:
                entity_text = entity.get('text', '')
                entity_type = entity.get('type', 'UNKNOWN')
                
                if not entity_text:
                    continue
                
                question = generate_question(
                    tokenizer, model, device,
                    src, entity_text, entity_type
                )
                
                questions.append({
                    "question": question,
                    "entity_type": entity_type,
                    "entity_text": entity_text
                })
                total_questions += 1
            
            output_row = {
                'src': src,
                'lang_tgt': data.get('lang_tgt', ''),
                'entities': entities,
                'questions': questions,
                'row_indexes': row_indexes
            }
            
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*50}")
    print(f"Entity-Aware QG Complete")
    print(f"Processed: {processed} samples")
    print(f"Total questions generated: {total_questions}")
    print(f"Output: {args.output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
