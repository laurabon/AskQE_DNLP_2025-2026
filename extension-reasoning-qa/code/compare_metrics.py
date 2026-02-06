"""
Compare Metrics: Baseline QA vs Reasoning QA

Compares the test results from the reasoning extension with the baseline.
Computes F1, EM, and shows ΔASKQE for each sample.
"""

import json
import os
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize text for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def parse_answers(answers_str: str) -> list:
    """Parse answers from string to list."""
    if isinstance(answers_str, list):
        return answers_str
    try:
        return json.loads(answers_str)
    except:
        return [answers_str]


def calculate_askqe(source_answers: list, bt_answers: list) -> dict:
    """Calculate ASKQE metrics for a single example."""
    if not source_answers or not bt_answers:
        return None
    
    min_len = min(len(source_answers), len(bt_answers))
    f1_scores = []
    em_scores = []
    
    for i in range(min_len):
        src = str(source_answers[i]) if source_answers[i] else ""
        bt = str(bt_answers[i]) if bt_answers[i] else ""
        
        if src.strip() and bt.strip():
            f1_scores.append(f1_score(bt, src))
            em_scores.append(exact_match(bt, src))
    
    if not f1_scores:
        return None
    
    return {
        "f1": sum(f1_scores) / len(f1_scores),
        "em": sum(em_scores) / len(em_scores),
        "num_questions": len(f1_scores)
    }


def main():
    print("=" * 70)
    print("ASKQE COMPARISON: Baseline vs Reasoning")
    print("=" * 70)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(extension_dir)
    
    # Load reasoning test results
    test_file = os.path.join(extension_dir, "results", "test", "test_es_atomic_alteration.jsonl")
    
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        return
    
    # Load baseline results
    baseline_source_file = os.path.join(project_root, "results Qwen3B baseline", "QA", "source", "en-atomic.jsonl")
    baseline_bt_file = os.path.join(project_root, "results Qwen3B baseline", "QA", "bt", "es", "atomic", "es-atomic-alteration.jsonl")
    
    # Build baseline lookup by ID
    baseline_source = {}
    baseline_bt = {}
    
    if os.path.exists(baseline_source_file):
        with open(baseline_source_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                baseline_source[data.get('id')] = parse_answers(data.get('answers', []))
    
    if os.path.exists(baseline_bt_file):
        with open(baseline_bt_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                baseline_bt[data.get('id')] = parse_answers(data.get('answers', []))
    
    print(f"\nBaseline entries: source={len(baseline_source)}, bt={len(baseline_bt)}")
    
    # Compare
    print("\n" + "-" * 70)
    print(f"{'ID':<15} {'Baseline F1':>12} {'Reasoning F1':>13} {'Δ F1':>10} {'Status'}")
    print("-" * 70)
    
    results = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sample_id = data.get('id')
            
            # Reasoning QA results
            reasoning_source = parse_answers(data.get('source_answers', []))
            reasoning_bt = parse_answers(data.get('bt_answers', []))
            
            # Baseline QA results
            base_source = baseline_source.get(sample_id, [])
            base_bt = baseline_bt.get(sample_id, [])
            
            # Calculate ASKQE for both
            baseline_metrics = calculate_askqe(base_source, base_bt)
            reasoning_metrics = calculate_askqe(reasoning_source, reasoning_bt)
            
            if baseline_metrics and reasoning_metrics:
                delta_f1 = reasoning_metrics["f1"] - baseline_metrics["f1"]
                
                # Determine status
                if delta_f1 > 0.05:
                    status = "✅ Improved"
                elif delta_f1 < -0.05:
                    status = "⚠️ Decreased"
                else:
                    status = "≈ Similar"
                
                print(f"{sample_id:<15} {baseline_metrics['f1']:>12.4f} {reasoning_metrics['f1']:>13.4f} {delta_f1:>+10.4f} {status}")
                
                results.append({
                    "id": sample_id,
                    "baseline_f1": baseline_metrics["f1"],
                    "reasoning_f1": reasoning_metrics["f1"],
                    "delta_f1": delta_f1,
                    "baseline_em": baseline_metrics["em"],
                    "reasoning_em": reasoning_metrics["em"]
                })
            else:
                print(f"{sample_id:<15} {'N/A':>12} {'N/A':>13} {'N/A':>10}")
    
    # Summary
    if results:
        avg_baseline = sum(r["baseline_f1"] for r in results) / len(results)
        avg_reasoning = sum(r["reasoning_f1"] for r in results) / len(results)
        avg_delta = sum(r["delta_f1"] for r in results) / len(results)
        
        print("-" * 70)
        print(f"{'AVERAGE':<15} {avg_baseline:>12.4f} {avg_reasoning:>13.4f} {avg_delta:>+10.4f}")
        print("=" * 70)
        
        print("\n📊 ANALYSIS:")
        if avg_delta > 0.05:
            print("   Reasoning QA shows HIGHER answer overlap → possible over-stabilization")
        elif avg_delta < -0.05:
            print("   Reasoning QA shows LOWER answer overlap → more sensitive to errors")
        else:
            print("   Minimal difference between methods")
        
        # Count improved/decreased
        improved = sum(1 for r in results if r["delta_f1"] > 0.05)
        decreased = sum(1 for r in results if r["delta_f1"] < -0.05)
        similar = len(results) - improved - decreased
        
        print(f"\n   Improved: {improved}/{len(results)}")
        print(f"   Decreased: {decreased}/{len(results)}")
        print(f"   Similar: {similar}/{len(results)}")
        
        # Save comparison results
        output_file = os.path.join(extension_dir, "results", "test", "comparison_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {"lang": "es", "pipeline": "atomic", "perturbation": "alteration"},
                "summary": {
                    "avg_baseline_f1": avg_baseline,
                    "avg_reasoning_f1": avg_reasoning,
                    "avg_delta_f1": avg_delta,
                    "num_samples": len(results)
                },
                "per_sample": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
