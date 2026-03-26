"""Filter data files to only include rows with incorrect_test_cases_shown data_source."""
import pandas as pd
import glob
import os

DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SOURCE = "coding/test_cases_parent_dir/incorrect_test_cases_shown/reward_check_function"

for base in ["data400", "data400_prefixed"]:
    for ext in [".parquet", ".jsonl"]:
        src = os.path.join(DIR, f"{base}{ext}")
        dst = os.path.join(DIR, f"incorrect_only_{base}{ext}")
        if not os.path.exists(src):
            print(f"Skipping {src} (not found)")
            continue

        if ext == ".parquet":
            df = pd.read_parquet(src)
        else:
            df = pd.read_json(src, lines=True)

        filtered = df[df["data_source"] == TARGET_SOURCE].reset_index(drop=True)
        print(f"{os.path.basename(src)}: {len(df)} -> {len(filtered)} rows")

        if ext == ".parquet":
            filtered.to_parquet(dst)
        else:
            filtered.to_json(dst, lines=True, orient="records")

print("Done.")
