# Judge LLM Fine-Tuning (Llama-3.2-3B-Instruct + QLoRA)

This project fine-tunes a **safety judge LLM** that returns only JSON:

- `verdict`: `ALLOW | DENY | ASK_CLARIFY`
- `confidence`: float in `[0, 1]`
- `violations`: list of `{rule_id, evidence}`
- `notes`: short explanation (<= 1 sentence)

The fine-tuned model is a **LoRA adapter** on top of:

- Base model: `meta-llama/Llama-3.2-3B-Instruct`

---

## 1) What We Built

We trained a judge model using:

- **QLoRA (4-bit NF4)** for memory-efficient tuning on a 12 GB GPU
- **LoRA adapters** (train small low-rank matrices, keep base weights frozen)
- **TRL SFTTrainer** with **prompt/completion** format

Primary scripts:

- Training: [train_llama32_3b_judge_qlora_wsl.py](d:/Projects/JudgeLLM-Finetuning/train_llama32_3b_judge_qlora_wsl.py)
- Evaluation (base vs adapter): [evaluate_base_vs_adapter.py](d:/Projects/JudgeLLM-Finetuning/evaluate_base_vs_adapter.py)

---

## 2) Environment

Validated setup:

- WSL2 Ubuntu
- GPU: RTX 4070 Laptop GPU (12 GB)
- Python: 3.11.14
- CUDA: 12.4
- torch: 2.6.0+cu124
- transformers: 5.2.0
- trl: 0.28.0
- peft: 0.18.1
- bitsandbytes: installed in venv

### Important dependency fix

Training initially failed due to missing `Python.h` during Triton/bitsandbytes import.  
Fix:

```bash
sudo apt update
sudo apt install -y python3.11-dev python3-dev build-essential gcc g++
```

---

## 3) Dataset and Formatting

Dataset location:

- `judge_sft_llama4scout_1k/judge_sft_llama4scout_1k/`
  - `train.jsonl` (800)
  - `dev.jsonl` (100)
  - `test.jsonl` (100)

Per-row schema includes:

- `id`, `source`, `instruction`
- `expected_verdict`, `expected_rules`
- `messages[]` with chat roles (`system`, `user`, `assistant`)

### Why prompt/completion split

Recent TRL flow for SFT expects supervised signal on response tokens (completion side).  
So we convert:

- `prompt`: chat up to assistant turn (`apply_chat_template(..., add_generation_prompt=True)`)
- `completion`: final assistant JSON + EOS

This is implemented in `to_prompt_completion(...)` inside the training script.

---

## 4) Training Configuration

Key config used:

- `MAX_LEN=512`
- `EPOCHS=2`
- `LR=2e-4`
- `GRAD_ACCUM=15`
- `SAVE_STEPS=100`
- `EVAL_STEPS=100`
- `WARMUP_FRAC=0.05`

Derived values:

- train samples = 800
- effective batch size = `1 * 15 = 15`
- steps/epoch = `ceil(800/15) = 54`
- total steps = `54 * 2 = 108`
- warmup steps = `int(0.05 * 108) = 5`

Memory/speed settings:

- 4-bit quantization (`nf4`, double quant)
- `optim="paged_adamw_8bit"`
- gradient checkpointing enabled
- mixed precision auto-select:
  - BF16 on Ampere+ GPUs
  - FP16 fallback otherwise

---

## 5) What Got Saved

Output directory:

- `outputs/llama32_3b_judge_qlora/`

Important artifacts:

- `adapter_final/` (final LoRA adapter for inference)
  - `adapter_model.safetensors`
  - `adapter_config.json`
  - tokenizer files
- `checkpoint-100/`, `checkpoint-108/`
  - adapter + optimizer + scheduler + trainer state

Resume behavior:

- Script checks `get_last_checkpoint(OUT_DIR)` and resumes automatically if present.

---

## 6) Evaluation Pipeline (Base vs Adapter)

Script:

- [evaluate_base_vs_adapter.py](d:/Projects/JudgeLLM-Finetuning/evaluate_base_vs_adapter.py)

What it does:

1. Loads base model in 4-bit
2. Runs test set and collects predictions
3. Loads adapter via `PeftModel.from_pretrained(base_model, adapter_dir)`
4. Runs same test set
5. Produces:
   - `comparison_summary.md`
   - `comparison_table.csv`
   - `metrics_base.json`
   - `metrics_adapter.json`
   - per-sample predictions JSON files

Run:

```bash
python evaluate_base_vs_adapter.py
```

---

## 7) Results

From `outputs/llama32_3b_judge_qlora/eval/comparison_summary.md` (test=100):

| metric | base_model | adapter_model | delta(adapter-base) |
|---|---:|---:|---:|
| samples | 100 | 100 | 0 |
| json_valid_rate | 0.9400 | 1.0000 | +0.0600 |
| verdict_accuracy | 0.6100 | 0.9000 | +0.2900 |
| rules_exact_match | 0.3400 | 0.7800 | +0.4400 |
| deny_recall | 0.7750 | 0.9500 | +0.1750 |
| unsafe_false_negative_rate_deny | 0.2250 | 0.0500 | -0.1750 |

Interpretation:

- Strong gains in verdict correctness and rule extraction.
- Large reduction in unsafe misses (`DENY` false negatives).
- Format control improved (higher JSON validity).

---

## 8) Practical Gotchas We Hit

1. **Missing Python headers in WSL** caused Triton/bitsandbytes compile error:
   - symptom: `fatal error: Python.h: No such file or directory`
2. **TRL API mismatch**:
   - `max_length`/`packing` must be set in `SFTConfig` (not as direct `SFTTrainer` kwargs in this setup).
3. **Mixed precision mismatch**:
   - BF16 gradients with FP16 AMP scaler can crash.
   - fixed by aligning model dtype and trainer `bf16/fp16` flags.
4. **Transformers deprecation**:
   - use `dtype=` instead of `torch_dtype=`.

---

## 9) Current Limitations

- Evaluation is on a relatively small test set (`n=100`).
- Current JSON-valid metric can still be optimistic if parser recovers inner JSON from noisy text.
- In-distribution performance may not fully reflect adversarial or out-of-distribution safety behavior.

---

## 10) Recommended Next Steps

1. Add strict schema validation metric (`schema_valid_rate`).
2. Evaluate on adversarial prompt-injection/jailbreak prompts.
3. Run OOD hazard scenarios and class-wise breakdown.
4. Check confidence calibration (`confidence` vs actual correctness).
5. Expand test size for tighter confidence in metrics.

---

## 11) Quick Repro Commands

Train:

```bash
python train_llama32_3b_judge_qlora_wsl.py
```

Evaluate:

```bash
python evaluate_base_vs_adapter.py
```

