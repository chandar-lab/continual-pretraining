import os
import argparse

#get args for min and max file number
parser = argparse.ArgumentParser()

parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)
parser.add_argument('--jsonl-key', type=str, default="text")
parser.add_argument('--start', type=float,)
parser.add_argument('--end', type=float,)

parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--merge', action='store_true')
parser.add_argument('--dir', type=str, default=None)
args = parser.parse_args()

jsonl_files = sorted([x for x in os.listdir(args.input_path) if '.jsonl' in x])

# if not args.dryrun:
print("Making dir: {}".format(args.output_path))
os.makedirs(args.output_path, exist_ok=True)

print(len(jsonl_files))

start = int(args.start / 100 * len(jsonl_files))
end = int(args.end / 100 * len(jsonl_files))

print("using start and end: {} {}".format(start, end))

for file in jsonl_files[start:end]:
    command = 'python tools/datasets/preprocess_data.py \
        --input \"{}\" \
        --output-prefix \"{}\" \
        --vocab /ccs/home/btherien/bif151/proj-shared/Meta-Llama-3-8B/tokenizer.json \
        --tokenizer-type Llama3HFTokenizer \
        --append-eod \
        --jsonl-keys {} \
        --workers 32'.format(
                os.path.join(args.input_path, file),
                os.path.join(args.output_path,file.replace('.parquet','').replace('.jsonl','')),
                args.jsonl_key
            )
    print(command)
    if not args.dryrun:
        os.system(command)

exit(0)



#llama3: /ccs/home/btherien/bif151/proj-shared/Meta-Llama-3-8B/tokenizer.json
#llama2: 