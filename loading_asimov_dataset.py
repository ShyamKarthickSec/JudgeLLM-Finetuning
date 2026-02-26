import os
import json
import tensorflow_datasets as tfds
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================

BASE_DIR = r"D:\Projects\JudgeLLM-Finetuning"
TFDS_DIR = os.path.join(BASE_DIR, "tfds_data")
EXPORT_DIR = os.path.join(BASE_DIR, "processed_data")

DATASET_NAME = "asimov_v2_injuries"
SPLIT = "train"

# ===============================
# CREATE DIRECTORIES
# ===============================

os.makedirs(TFDS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"Downloading dataset into: {TFDS_DIR}")

# ===============================
# DOWNLOAD + PREPARE
# ===============================

builder = tfds.builder(DATASET_NAME, data_dir=TFDS_DIR)
builder.download_and_prepare()

print("\nDataset Info:")
print(builder.info)

# ===============================
# LOAD DATASET
# ===============================

dataset = builder.as_dataset(split=SPLIT)

# ===============================
# EXPORT TO JSONL (LLM-friendly)
# ===============================

output_file = os.path.join(EXPORT_DIR, f"{DATASET_NAME}_{SPLIT}.jsonl")

print(f"\nExporting to: {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    for example in tqdm(tfds.as_numpy(dataset)):
        json.dump(
            {k: v.tolist() if hasattr(v, "tolist") else v.decode("utf-8") if isinstance(v, bytes) else v
             for k, v in example.items()},
            f
        )
        f.write("\n")

print("\nDownload and export complete.")