import sys
sys.path.append('/ccs/home/btherien/bif151/scratch/btherien/neox/gpt-neox')
from megatron.data.indexed_dataset import MMapIndexedDataset

import numpy as np
import json
import time
import os
import sys
import multiprocessing
from tqdm import tqdm
from os.path import join as j

import argparse


class SilenceStdout:
    def __enter__(self):
        self.saved_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, *args):
        sys.stdout = self.saved_stdout

def get_num_tokens(path_to_dataset):
    with SilenceStdout():
        ds = MMapIndexedDataset(path_to_dataset)
        return int(np.sum(ds._index._sizes))



def get_tokens_in_dataset_mp(data_path, output_dir, config_prefix='train', num_proc=multiprocessing.cpu_count()):

    def process_file(f):
        dataset_path = os.path.join(data_path, dir_name, f[:-len('.bin')])
        return dataset_path, get_num_tokens(dataset_path)

    dirs = os.listdir(data_path)
    dataset_map = {}
    for dir_name in dirs:
        start_time = time.time()  # start timing
        print("Starting dir:", dir_name, "...")
        pool = multiprocessing.Pool(num_proc)
        files = [x for x in os.listdir(os.path.join(data_path, dir_name)) if x.endswith('.bin')]
        inner_map = dict(pool.map(process_file, files))
        pool.close()
        pool.join()
        dataset_map[dir_name] = inner_map

        end_time = time.time()  # end timing
        print(f"Time for {dir_name}: {end_time - start_time} seconds")

    print("done Looping")
    print(dataset_map)

    total = sum(dataset_map.values())

    # dataset_config = {
    #     "train-data-paths": [ [ y.replace('.bin','') for y in os.listdir(j(data_path,dir_name)) if y.endswidth('.bin') and 'merged' in y][0] for dir_name in dirs ]
    #     "train-data-weights": [ x / total for x in dataset_map.values()]
    # }

    # with open(f'{config_prefix}_dataset_config.json', 'w') as json_file:
    #     json.dump(dataset_map, json_file)

    # save the dictionary to a json file
    with open(j(data_path,'token_counts.json'), 'w') as json_file:
        json.dump(dataset_map, json_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing all document files to merge",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Path to binary output file without suffix",
    )
    parser.add_argument(
        "--config-prefix",
        type=str,
        # options=['train','test','val'],
        required=True,
        help="Path to binary output file without suffix",
    )

    args = parser.parse_args()

    get_tokens_in_dataset_mp(args.data_path, args.data_path, config_prefix=args.config_prefix, num_proc=4)