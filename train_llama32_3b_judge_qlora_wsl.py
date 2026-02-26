import os
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# -----------------------
# CONFIG (EDIT PATHS)
# -----------------------
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

DATA_DIR = "/mnt/d/Projects/JudgeLLM-Finetuning/judge_sft_llama4scout_1k/judge_sft_llama4scout_1k"   # <-- change if needed
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
DEV_FILE   = os.path.join(DATA_DIR, "dev.jsonl")
TEST_FILE  = os.path.join(DATA_DIR, "test.jsonl")

OUT_DIR = "/mnt/d/Projects/JudgeLLM-Finetuning/outputs/llama32_3b_judge_qlora"

MAX_LEN = 512   # if OOM -> 512
EPOCHS = 2
LR = 2e-4
GRAD_ACCUM = 15
SAVE_STEPS = 100
EVAL_STEPS = 100
WARMUP_FRAC = 0.05

# -----------------------
# Load dataset
# -----------------------
data_files = {"train": TRAIN_FILE, "validation": DEV_FILE, "test": TEST_FILE}
ds = load_dataset("json", data_files=data_files)

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.model_max_length = MAX_LEN
tokenizer.truncation_side = "right"

def to_prompt_completion(example):
    """
    TRL 0.28 expects dataset columns:
      - prompt
      - completion
    We'll split the chat messages accordingly.
    """
    msgs = example["messages"]

    if not msgs:
        return {"prompt": "", "completion": tokenizer.eos_token}

    if msgs[-1].get("role") == "assistant":
        prompt_msgs = msgs[:-1]
        completion = msgs[-1].get("content", "")
    else:
        # Fallback: no assistant at end
        prompt_msgs = msgs
        completion = ""

    # Prompt: add_generation_prompt=True makes template end expecting assistant
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    completion_text = completion + tokenizer.eos_token
    return {"prompt": prompt_text, "completion": completion_text}

# Keep only prompt/completion to reduce RAM
ds_pc = ds.map(
    to_prompt_completion,
    remove_columns=ds["train"].column_names,
    desc="Formatting -> prompt/completion",
)

print("Sample PROMPT:\n", ds_pc["train"][0]["prompt"][:500])
print("\nSample COMPLETION:\n", ds_pc["train"][0]["completion"][:200])

# -----------------------
# Model: QLoRA 4-bit
# -----------------------
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=compute_dtype,
)

model.config.use_cache = False

# -----------------------
# LoRA config
# -----------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

# -----------------------
# Training args (Transformers 5.2 uses eval_strategy, not evaluation_strategy)
# Compute warmup_steps from fraction
# -----------------------
num_train = len(ds_pc["train"])
per_device_bs = 1
steps_per_epoch = math.ceil(num_train / (per_device_bs * GRAD_ACCUM))
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

print(f"\nnum_train={num_train} steps/epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}\n")
print(f"mixed_precision: {'bf16' if use_bf16 else 'fp16'}")

args = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=per_device_bs,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_steps=warmup_steps,
    logging_steps=20,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,

    fp16=not use_bf16,
    bf16=use_bf16,

    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",

    gradient_checkpointing=True,
    report_to="none",
    max_length=MAX_LEN,
    packing=False,
    completion_only_loss=True,
)

# -----------------------
# Trainer (TRL 0.28 uses processing_class)
# -----------------------
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds_pc["train"],
    eval_dataset=ds_pc["validation"],
    peft_config=lora_config,
    processing_class=tokenizer,
)

# -----------------------
# Train (auto-resume)
# -----------------------
os.makedirs(OUT_DIR, exist_ok=True)
last_ckpt = get_last_checkpoint(OUT_DIR)
if last_ckpt:
    print(f"[resume] checkpoint found: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("[start] fresh training run")
    trainer.train()

# -----------------------
# Save adapter
# -----------------------
adapter_dir = os.path.join(OUT_DIR, "adapter_final")
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("[save] adapter saved to:", adapter_dir)

