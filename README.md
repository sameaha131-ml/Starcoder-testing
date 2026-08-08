# Starcoder2-3B Code Repair Fine-tuning

Fine-tuned `bigcode/starcoder2-3b` with QLoRA on a Python bug-fixing dataset.

## Setup
- Base model: bigcode/starcoder2-3b
- Fine-tuning method: QLoRA (4-bit NF4, r=16, alpha=16)
- Max sequence length: 512 tokens
- Hardware: 2x Tesla T4 (15GB each)
- Framework: PyTorch, HuggingFace Transformers, PEFT, bitsandbytes

## Dataset
Custom hybrid dataset of Python buggy/fixed code pairs.
- Train: 20,077 samples
- Validation: 2,510 samples
- Test: 2,510 samples

## Checkpoint Selection
Best checkpoint: step 2500 (lowest eval loss on 300-sample validation subset).

| Checkpoint | Eval Loss |
|------------|-----------|
| 500 | 0.119273 |
| 1000 | 0.111461 |
| 1500 | 0.108638 |
| 2000 | 0.105116 |
| **2500** | **0.102107** |
| 3000 | 0.103808 |
| 3500 | 0.103310 |
| 3765 | 0.103318 |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Exact Match Rate | 0.00% |
| Corpus BLEU | 13.90 |
| Per-sample BLEU (mean) | 15.69 |
| Per-sample BLEU (median) | 6.14 |
| ROUGE-1 (mean) | 24.46 |
| ROUGE-1 (median) | 12.75 |
| ROUGE-2 (mean) | 22.74 |
| ROUGE-2 (median) | 10.96 |
| ROUGE-L (mean) | 24.27 |
| ROUGE-L (median) | 12.67 |
| % BLEU=100 | 0.0% |
| % BLEU=0 | 1.7% |

## Notes
- Model tends to generate additional code beyond the exact fix (hallucination).
- 23% of predictions still contain prompt template fragments after extraction.
- Exact match is 0% — model rarely reproduces the ground truth exactly.
- Better results may require: different loss masking, longer training, or inference-time stopping criteria.

## Files
- `evaluation_results.csv`: Per-sample predictions with all metrics (exact match, BLEU, ROUGE-1/2/L)
- `checkpoint_ranking.csv`: Eval loss across all checkpoints
- `predictions_v4_full.csv`: Raw model outputs for all test samples
