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
Best checkpoint chosen by lowest evaluation loss on a 300-sample validation subset.

## Evaluation Results
| Metric | Value |
|--------|-------|
| Test Loss |     |
| Exact Match Rate |     |
| BLEU Score |     |

## Files
- `evaluation_results.csv`: Per-sample predictions and metrics
- `checkpoint_ranking.csv`: Loss across all checkpoints

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")
model = PeftModel.from_pretrained(model, "path/to/best_checkpoint")
```


Replace `[ADD]` with actual numbers after evaluation. Want me to include anything else?# Starcoder-testing
