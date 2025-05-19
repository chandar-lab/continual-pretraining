import os
import os.path as osp
import json
from tqdm import tqdm
from datasets import Dataset
from multiprocessing import Pool

arrow_path = '/lustre/orion/bif151/proj-shared/japaneese_ista/kajuma___abeja-cc-ja/default/0.0.0/fecbcb63aa6772df386262ed934b3f0db23339fa'
jsonl_path = '/lustre/orion/bif151/proj-shared/japanese_ista/japanese_jsonl'
os.makedirs(jsonl_path, exist_ok=True)

arrow_files = [f for f in os.listdir(arrow_path) if f.endswith('.arrow')]

def convert_arrow_to_jsonl(arr_name):
    arrow_filepath = osp.join(arrow_path, arr_name)
    output_file = osp.join(jsonl_path, arr_name.replace('.arrow', '.jsonl'))
    dataset = Dataset.from_file(arrow_filepath)
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    return arr_name

if __name__ == "__main__":
    with Pool(processes=16) as pool:
        # Wrap with tqdm for progress bar over map
        for _ in tqdm(pool.imap_unordered(convert_arrow_to_jsonl, arrow_files), total=len(arrow_files)):
            pass
