import os
import shutil
import re

# source_dir = "/lustre/orion/bif151/proj-shared/arabic/cluster_jsonl_tokenized"
# test_dir = "/lustre/orion/bif151/proj-shared/arabic/cluster_jsonl_tokenized_test"

# os.makedirs(test_dir, exist_ok=True)

# # List all .idx files sorted naturally
# idx_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".idx")])

# # Extract shard numbers from idx filenames using regex
# def shard_num(filename):
#     # Example: 101_billion_arabic_words_dataset-train-00470-of-00474_text_document.idx
#     m = re.search(r"-(\d+)-of-\d+_", filename)
#     return int(m.group(1)) if m else -1

# idx_files_sorted = sorted(idx_files, key=shard_num)

# # Get last 10 shard numbers
# last_10_shards = [shard_num(f) for f in idx_files_sorted[-10:]]

# for shard in last_10_shards:
#     # Build expected .bin and .idx filenames for this shard
#     bin_name = None
#     idx_name = None
#     for f in os.listdir(source_dir):
#         if f"-{shard:05d}-of-" in f:
#             if f.endswith(".bin"):
#                 bin_name = f
#             elif f.endswith(".idx"):
#                 idx_name = f
#     # Copy both if they exist
#     if bin_name:
#         shutil.copy2(os.path.join(source_dir, bin_name), os.path.join(test_dir, bin_name))
#         print(f"Copied {bin_name}")
#     if idx_name:
#         shutil.copy2(os.path.join(source_dir, idx_name), os.path.join(test_dir, idx_name))
#         print(f"Copied {idx_name}")


# import os
# import shutil
# import re

# source_dir = "/lustre/orion/bif151/proj-shared/japanese_ista/japanese_jsonl"
# test_dir = "/lustre/orion/bif151/proj-shared/japanese_ista/japanese_jsonl_test"

# os.makedirs(test_dir, exist_ok=True)


# jsonl_files = [f for f in os.listdir(source_dir) if f.endswith(".jsonl")]
# print(f"Found {len(jsonl_files)} .jsonl files in source_dir")

# def shard_num(filename):
#     m = re.search(r"-(\d{5})-of-\d+", filename)
#     if m:
#         return int(m.group(1))
#     else:
#         print(f"WARNING: Could not extract shard number from {filename}")
#         return -1

# jsonl_files_sorted = sorted(jsonl_files, key=shard_num)
# print("Top 5 sorted files by shard_num:")
# for f in jsonl_files_sorted[:5]:
#     print(f"  {f} -> shard {shard_num(f)}")

# last_10_shards = [shard_num(f) for f in jsonl_files_sorted[-10:]]
# print(f"Last 10 shard numbers: {last_10_shards}")

# for shard in last_10_shards:
#     moved = False
#     # Use zero-padded format here
#     shard_str = f"{shard:05d}"
#     for f in os.listdir(source_dir):
#         if f.endswith(".jsonl") and f"-{shard_str}-of-" in f:
#             src_path = os.path.join(source_dir, f)
#             dst_path = os.path.join(test_dir, f)
#             shutil.move(src_path, dst_path)
#             print(f"Moved {f}")
#             moved = True
#     if not moved:
#         print(f"No .jsonl file found for shard {shard_str}")











import os
import shutil
import re

source_dir = "/lustre/orion/bif151/proj-shared/japanese_ista/japanese_train_tokenized"
dest_dir   = "/lustre/orion/bif151/proj-shared/japanese_ista/japanese_train_tokenized_part2"

os.makedirs(dest_dir, exist_ok=True)

# regex to pull out the shard number, e.g. “-01043-of-02098_”
pattern = re.compile(r"-(\d+)-of-\d+_")

for fname in os.listdir(source_dir):
    m = pattern.search(fname)
    if not m:
        continue
    shard = int(m.group(1))
    if shard > 1043:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_dir, fname)
        shutil.move(src, dst)
        print(f"Moved {fname}")
