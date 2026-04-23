# Red Team Judge — Fine-Tuning on Apple Silicon

> Fine-tune a safety judge LLM in under 30 minutes on an 8GB MacBook. No cloud. No GPU. No $10k bill.

A complete, beginner-friendly pipeline for fine-tuning a small LLM to act as an **AI safety judge** using LoRA on Apple Silicon.

---

## What Does This Do?

Trains `Qwen2.5-1.5B-Instruct` to automatically evaluate whether AI responses are **safe or harmful**. The fine-tuned model outputs structured verdicts with reasoning:

```json
{
  "verdict": "unsafe",
  "category": "dangerous_advice",
  "severity": "high",
  "reasoning": "Recommends extreme caloric restriction and laxative abuse."
}
```

It can detect: `violence`, `self_harm`, `hate_speech`, `misinformation`, `manipulation`, `dangerous_advice`, `privacy_violation`, `illegal_activity`, `sexual_content`, `bias`

And it knows the difference between *talking about* sensitive topics (safe) vs. *promoting harm* (unsafe).

---

## Hardware Requirements

| Requirement | Minimum |
|-------------|---------|
| Chip | Apple Silicon (M1/M2/M3) |
| RAM | 8 GB |
| Disk | ~5 GB free (for model download) |
| Python | 3.8+ |

> Works on any M-series MacBook — Air, Pro, or Mac mini.

---

## Quick Start

```bash
# 1. Install dependency
pip install -r requirements.txt

# 2. Generate training data
python prepare_data.py

# 3. Fine-tune (15–30 min on M1 Pro)
python run_finetune.py

# 4. Test the model
python run_inference.py

# 5. Evaluate on validation set
python run_inference.py --evaluate

# 6. Try your own examples
python run_inference.py --interactive
```

---

## Project Structure

```
red-team-judge/
│
├── prepare_data.py        ← Generates 100 labeled safety examples → train/valid JSONL
├── lora_config.yaml       ← All LoRA hyperparameters (rank, alpha, lr, iters...)
├── run_finetune.py        ← Full training pipeline (data → download → train → test)
├── run_inference.py       ← Test, evaluate, and interact with the fine-tuned model
│
├── train.jsonl            ← 55 training examples (auto-generated)
├── valid.jsonl            ← 14 validation examples (auto-generated)
│
├── adapters/
│   ├── adapter_config.json          ← LoRA structure metadata
│   ├── adapters.safetensors         ← Final trained adapter weights (~5MB)
│   └── 0000050_adapters.safetensors ← Checkpoints at steps 50/100/150
│
├── requirements.txt       ← mlx-lm>=0.19.0
├── .gitignore
│
└── deep_dive.md           ← Complete technical explainer (every concept from scratch)
```

---

## How It Works

This project uses **LoRA (Low-Rank Adaptation)** — a technique that adapts a frozen pre-trained model by training only tiny "adapter" matrices (~5MB) instead of all 1.5 billion parameters. This makes it possible to fine-tune on a MacBook with 8GB of RAM.

```
Base model (Qwen2.5-1.5B):  3 GB  — frozen, never updated
LoRA adapters:               5 MB  — the only thing trained
Training time:            15-30 min on M1 Pro
```

For a complete explanation of every concept — transformers, layers, rank, alpha, loss, gradient checkpointing, precision/recall — read the **[deep dive explainer](deep_dive.md)**.

---

## Training Details

| Setting | Value | Why |
|---------|-------|-----|
| Base model | `Qwen2.5-1.5B-Instruct` | Small enough for 8GB, strong instruction-following |
| LoRA rank | 8 | Enough capacity for this narrow task |
| LoRA layers | 16 (of 28) | Task-specific layers only |
| Training steps | 200 | ~3.6 epochs over 55 examples |
| Learning rate | 1e-5 | Conservative — preserves pre-trained knowledge |
| Batch size | 1 | Memory constraint |
| Gradient checkpointing | Yes | Required to fit in 8GB |

---

## Customization

### Change the base model
Edit `BASE_MODEL` in `run_finetune.py` and `run_inference.py`:
```python
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"   # lighter alternative
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"        # 3.8B, tight on 8GB
```

### Train longer or with more capacity
Edit `lora_config.yaml`:
```yaml
iters: 500    # more training (default: 200)
rank: 16      # more adapter capacity (default: 8)
```

### Add more training examples
Edit the `examples` list in `prepare_data.py` and re-run:
```bash
python prepare_data.py && python run_finetune.py
```

---

## Understanding the Code

New to fine-tuning? Every concept in this project is explained from scratch in **[deep_dive.md](deep_dive.md)**:

- What are transformer layers and why does the model have 28 of them?
- How does LoRA work mathematically (with ASCII diagrams)?
- What does every hyperparameter in `lora_config.yaml` actually do?
- What is loss, and how do you know training is working?
- What do precision, recall, and F1 mean for a safety classifier?

---

## License

MIT — use it, fork it, build on it.
