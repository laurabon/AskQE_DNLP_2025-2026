# Reasoning-Augmented QA Extension for ASKQE

This extension studies the effect of reasoning-augmented QA on the stability and sensitivity of ASKQE under back-translation.

## Structure

```
extension-reasoning-qa/
├── code/
│   ├── prompt_reasoning.py      # Reasoning-augmented prompt
│   ├── qwen-3b-reasoning.py     # QA script
│   ├── string_comparison.py     # Evaluation (F1, EM, chrF, BLEU)
│   └── sbert.py                 # Semantic similarity evaluation
└── results/
    ├── QA/
    │   ├── source/              # Answers from source sentences
    │   └── bt/                  # Answers from backtranslations
    └── evaluation/
        ├── string-comparison/   # String comparison metrics
        └── sbert/               # SBERT cosine similarity
```

## Usage

### 1. Run Question Answering (requires GPU)

```bash
cd extension-reasoning-qa/code
python qwen-3b-reasoning.py --run_all
```

### 2. Run Evaluation

```bash
# String comparison
python string_comparison.py

# SBERT semantic similarity
python sbert.py
```

## Research Question

> Does reasoning-augmented QA improve the reliability of ASKQE, or does it reduce its sensitivity to translation errors by over-stabilizing answer pairs?

## Comparison with Baseline

Compare results in `extension-reasoning-qa/results/` with `results Qwen3B baseline/` to analyze the effect of reasoning.
