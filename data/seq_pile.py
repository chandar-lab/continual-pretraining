import numpy as np

# Paths to your dataset and idx files
bin_file_path = '/lustre/orion/bif151/scratch/istabrak/gpt-neox/data/slim_pajama/tokenized_train_0-100B/ArXiv/ArXiv.bin'
idx_file_path = '/lustre/orion/bif151/scratch/istabrak/gpt-neox/data/slim_pajama/tokenized_train_0-100B/ArXiv/ArXiv.idx'

# Reading the IDX file to get the offsets of each record
# Assuming IDX contains 64-bit integers representing byte offsets for records
record_offsets = np.fromfile(idx_file_path, dtype=np.int64)

# Determine the number of records in the dataset
num_records = len(record_offsets)
print(f"Number of records in the arXiv dataset: {num_records}")

# Function to read a specific record based on its index
def read_record(index):
    with open(bin_file_path, 'rb') as f:
        f.seek(record_offsets[index])  # Seek to the record position
        # Assuming fixed record size for simplicity; adjust as necessary
        record_size = 1024  # Replace with actual record size in bytes
        record_data = f.read(record_size)
        return record_data

# Example: Reading the first record
first_record = read_record(0)
print(f"First record data: {first_record[:100]}")  # Print first 100 bytes for preview

# Example: Reading the last record
last_record = read_record(num_records - 1)
print(f"Last record data: {last_record[:100]}")  # Print first 100 bytes for preview
