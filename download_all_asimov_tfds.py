import os
import tensorflow_datasets as tfds

BASE_DIR = r"D:\Projects\JudgeLLM-Finetuning"
TFDS_DIR = os.path.join(BASE_DIR, "tfds_data")
os.makedirs(TFDS_DIR, exist_ok=True)

print(f"TFDS data_dir: {TFDS_DIR}")

# 1) List all datasets available in your current TFDS installation
all_datasets = tfds.list_builders()

# 2) Filter datasets that start with "asimov"
asimov_datasets = sorted([d for d in all_datasets if d.startswith("asimov")])

print("\nFound ASIMOV datasets in your TFDS install:")
for d in asimov_datasets:
    print("  -", d)

print(f"\nTotal: {len(asimov_datasets)} dataset(s)\n")

# 3) Download + prepare each dataset
failed = []
for name in asimov_datasets:
    try:
        print(f"\n=== Downloading: {name} ===")
        builder = tfds.builder(name, data_dir=TFDS_DIR)
        builder.download_and_prepare()
        print(f"✅ Done: {name}")
    except Exception as e:
        print(f"❌ Failed: {name}\n   Reason: {e}")
        failed.append((name, str(e)))

print("\n========================================")
print("Finished downloading ASIMOV datasets.")
if failed:
    print("\nSome datasets failed:")
    for n, msg in failed:
        print(f"  - {n}: {msg}")
else:
    print("\n✅ All ASIMOV datasets downloaded successfully.")