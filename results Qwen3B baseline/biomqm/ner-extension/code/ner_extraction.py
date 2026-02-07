"""
NER Extraction Module for BioMQM

Uses BioBERT for biomedical entity extraction (DISEASE, DRUG, etc.)

Usage:
    python ner_extraction.py --input_path /path/to/data.jsonl --output_path /path/to/output.jsonl
"""

import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# BioBERT NER model for biomedical entities
MODEL_NAME = "alvaroalon2/biobert_diseases_ner"


def load_ner_pipeline():
    """Load the NER pipeline."""
    print(f"Loading NER model: {MODEL_NAME}")
    
    device = 0 if torch.cuda.is_available() else -1
    
    ner = pipeline(
        "ner",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        aggregation_strategy="simple",
        device=device
    )
    
    print(f"NER model loaded on {'GPU' if device == 0 else 'CPU'}")
    return ner


def extract_entities(ner_pipeline, text):
    """
    Extract entities from text.
    
    Args:
        ner_pipeline: HuggingFace NER pipeline
        text: Input text string
        
    Returns:
        List of entities: [{"text": "...", "type": "...", "start": int, "end": int, "score": float}]
    """
    if not text or not text.strip():
        return []
    
    try:
        results = ner_pipeline(text)
        
        entities = []
        for entity in results:
            entities.append({
                "text": entity.get("word", ""),
                "type": entity.get("entity_group", "UNKNOWN"),
                "start": entity.get("start", 0),
                "end": entity.get("end", 0),
                "score": round(entity.get("score", 0.0), 4)
            })
        
        return entities
        
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="NER Extraction for BioMQM")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSONL file (dev_with_backtranslation.jsonl)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file with entities")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return
    
    # Load NER model
    ner = load_ner_pipeline()
    
    # Process file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"\nProcessing: {args.input_path}")
    
    # First pass: collect unique src sentences
    unique_src = {}  # src -> {data, row_indexes}
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                src = data.get('src', '')
                
                if src not in unique_src:
                    unique_src[src] = {
                        'src': src,
                        'lang_tgt': data.get('lang_tgt', ''),
                        'row_indexes': [idx]
                    }
                else:
                    unique_src[src]['row_indexes'].append(idx)
            except json.JSONDecodeError:
                continue
    
    total_unique = len(unique_src)
    print(f"Found {total_unique} unique source sentences")
    
    if args.max_samples:
        items = list(unique_src.items())[:args.max_samples]
    else:
        items = list(unique_src.items())
    
    # Second pass: extract entities
    processed = 0
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for src, data in items:
            processed += 1
            
            if processed % 50 == 0:
                print(f"[{processed}/{len(items)}] Processing...")
            
            # Extract entities
            entities = extract_entities(ner, src)
            
            output_row = {
                'src': src,
                'lang_tgt': data['lang_tgt'],
                'entities': entities,
                'row_indexes': data['row_indexes']
            }
            
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*50}")
    print(f"NER Extraction Complete")
    print(f"Processed: {processed} unique sentences")
    print(f"Output: {args.output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
