import os
from datasets import load_dataset
import datasets

print("starting download..")

if __name__ == "__main__":
    cache_dir="/lustre/orion/bif151/proj-shared/japaneese_ista/"
    BATCH_SIZE = 10
    NUM_PROC = 32
    datasets.config.DEFAULT_MAX_BATCH_SIZE = 10
    os.makedirs(cache_dir,exist_ok=True)
    ds = load_dataset("kajuma/ABEJA-CC-JA",
                        split="train",
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        writer_batch_size=BATCH_SIZE,
                        num_proc=NUM_PROC
                    )