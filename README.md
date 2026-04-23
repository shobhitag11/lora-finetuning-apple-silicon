# LoRA Fine-Tuning on Apple Silicon

> Fine-tune LLMs as AI safety judges on an 8GB MacBook. No cloud. No GPU. No $10k bill.

Two complete pipelines for LoRA fine-tuning small language models using [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) on Apple Silicon — one for Qwen 2.5 and one for Gemma 4.

---

## What this repo builds

Each pipeline fine-tunes a small model to act as an **AI safety judge**: given a user prompt and an AI response, it outputs a structured verdict:

```json
{
  "verdict": "unsafe",
  "category": "dangerous_advice",
  "severity": "high",
  "reasoning": "Recommends extreme caloric restriction and laxative abuse."
}
```

Detected harm categories: `violence`, `self_harm`, `hate_speech`, `misinformation`,
`manipulation`, `dangerous_advice`, `privacy_violation`, `illegal_activity`, `sexual_content`, `bias`

---

## Projects

| Directory | Model | RAM | Training time |
|-----------|-------|-----|---------------|
| [`qwen_fine_tuning/`](qwen_fine_tuning/) | Qwen2.5-1.5B-Instruct | 8 GB | 15–30 min |
| [`gemma_4_fine_tuning/`](gemma_4_fine_tuning/) | Gemma 4 E2B-IT (4-bit) | 8 GB | 10–30 min |

See each directory's README for model-specific setup and training details.

---

## Hardware requirements

- **Apple Silicon Mac** — M1, M2, M3, or M4 (any variant)
- **8 GB unified memory** minimum
- **~5 GB free disk** for model download (cached after first run)
- **Python 3.8+**

---

## Quick start

```bash
# Pick a project
cd qwen_fine_tuning        # or: cd gemma_4_fine_tuning

# Install dependency
pip install -r requirements.txt

# Generate training data
python prepare_data.py

# Fine-tune
python run_finetune.py

# Test the judge
python run_inference.py
python run_inference.py --interactive
python run_inference.py --evaluate
```

---

## How it works

Both pipelines use **LoRA (Low-Rank Adaptation)** to fine-tune only tiny adapter matrices (~5 MB) while the base model stays frozen. This makes it possible to train on a MacBook:

```
Base model:     1.5–2.3B params   frozen, ~3 GB
LoRA adapters:  ~5 MB             the only weights trained
Training time:  15–30 min         on M1 Pro 8GB
```

For a full explanation of every concept (transformers, LoRA, loss, gradient checkpointing, precision/recall), see [`qwen_fine_tuning/deep_dive.md`](qwen_fine_tuning/deep_dive.md).

---

## License

MIT
