import os
import argparse
import numpy as np

def merge_shards_streaming(input_dir, output_dir,
                           token_dtype=np.uint16,
                           index_dtype=np.int64,
                           bin_chunk_bytes=1024*1024,
                           idx_chunk_elems=1_000_000):
    os.makedirs(output_dir, exist_ok=True)

    # get sorted lists of shards
    bin_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".bin"))
    idx_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".idx"))
    assert len(bin_files) == len(idx_files), "mismatched bin/idx counts"

    merged_bin = os.path.join(output_dir, "merged_dataset.bin")
    merged_idx = os.path.join(output_dir, "merged_dataset.idx")

    # 1) STREAM all .bin files into merged_dataset.bin
    with open(merged_bin, "wb") as fout:
        for b in bin_files:
            path = os.path.join(input_dir, b)
            with open(path, "rb") as fin:
                while True:
                    chunk = fin.read(bin_chunk_bytes)
                    if not chunk:
                        break
                    fout.write(chunk)

    # 2) STREAM through each .idx file, adjust by offset, append to merged_dataset.idx
    offset = 0
    bytes_per_token = np.dtype(token_dtype).itemsize
    idx_dtype = np.dtype(index_dtype)
    bytes_per_idx = idx_dtype.itemsize

    with open(merged_idx, "wb") as fout_idx:
        for b, i in zip(bin_files, idx_files):
            bin_path = os.path.join(input_dir, b)
            idx_path = os.path.join(input_dir, i)

            # how many tokens in this shard?
            shard_size = os.path.getsize(bin_path)
            n_tokens = shard_size // bytes_per_token

            # read & adjust idx in chunks of idx_chunk_elems
            with open(idx_path, "rb") as fin_idx:
                # read in byte-chunks that align with element boundaries
                chunk_bytes = idx_chunk_elems * bytes_per_idx
                while True:
                    raw = fin_idx.read(chunk_bytes)
                    if not raw:
                        break
                    arr = np.frombuffer(raw, dtype=index_dtype)
                    arr += offset
                    fout_idx.write(arr.tobytes())

            offset += n_tokens

    print(f"✅ Merged {len(bin_files)} shards →")
    print(f"   • {merged_bin}")
    print(f"   • {merged_idx}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True,
                   help="where your .bin/.idx shards live")
    p.add_argument("--output_dir", default=None,
                   help="where to put merged files; defaults to input_dir")
    args = p.parse_args()
    out = args.output_dir or args.input_dir
    merge_shards_streaming(args.input_dir, out)
