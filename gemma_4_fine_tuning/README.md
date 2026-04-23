# Red Team Judge — Gemma 4 E2B Fine-Tuning

Fine-tune Google's **Gemma 4 E2B** as an AI safety judge on your MacBook using LoRA + MLX.

## What this does

Trains Gemma 4 (Google's latest open model, Apache 2.0) to evaluate AI responses for safety. The fine-tuned model classifies prompt-response pairs as **safe** or **unsafe** across 10 harm categories, outputting structured JSON verdicts with reasoning.

## Hardware

- **MacBook M1/M2/M3/M4** (any variant)
- **8GB+ unified memory** (uses 4-bit quantized model)
- **~4GB disk** (first run downloads the model, cached afterward)

## Quick start

```bash
cd gemma4-red-team-judge

# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data (100 curated examples)
python prepare_data.py

# 3. Fine-tune (10-30 min on M1 Pro)
python run_finetune.py

# 4. Use the judge
python run_inference.py                    # Run test cases
python run_inference.py --interactive      # Chat mode
python run_inference.py --evaluate         # Eval on validation set
python run_inference.py --evaluate-test    # Eval on held-out test set
```

## Model selection guide

| Model | RAM needed | Quality | ID |
|-------|-----------|---------|-----|
| **Gemma 4 E2B 4-bit** | ~5GB | Good | `mlx-community/gemma-4-e2b-it-4bit` |
| Gemma 4 E4B 4-bit | ~7GB | Better | `mlx-community/gemma-4-e4b-it-4bit` |
| Gemma 4 E2B full | ~10GB | Best (E2B) | `google/gemma-4-e2b-it` |

Default is E2B 4-bit (fits 8GB). Override with `--model`:

```bash
python run_finetune.py --model mlx-community/gemma-4-e4b-it-4bit
```

## Project structure

```
gemma4-red-team-judge/
├── prepare_data.py     # 100 curated safety examples → JSONL
├── lora_config.yaml    # LoRA hyperparameters (tuned for 8GB)
├── run_finetune.py     # Training pipeline
├── run_inference.py    # Test, evaluate, interactive mode
├── requirements.txt
└── data/
    ├── train.jsonl     # 70% training split
    ├── valid.jsonl     # 15% validation split
    └── test.jsonl      # 15% held-out test split
```

## Why Gemma 4 E2B?

- **Apache 2.0** — fully open, commercial use allowed
- **2.3B effective params** — fits 8GB RAM at 4-bit
- **State-of-the-art per parameter** — outperforms much larger older models
- **Native system prompt support** — cleaner fine-tuning
- **MLX day-1 support** — optimized for Apple Silicon

## Customization

**More training data:** Edit `examples` in `prepare_data.py`, re-run `python prepare_data.py`.

**Different task:** Change `SYSTEM_PROMPT` and examples. The pipeline is task-agnostic.

**Better quality:** Increase `iters` (try 500), `rank` (try 16), or use the E4B model.

**Troubleshooting OOM:** Reduce `lora_layers` to 4, or close other apps.

## License

Apache 2.0 (same as Gemma 4). Training data is synthetic.
