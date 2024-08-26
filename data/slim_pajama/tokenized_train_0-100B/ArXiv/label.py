import numpy as np
from data.indexed_dataset import MMapIndexedDatasetBuilder

# Define file paths
bin_file = "example.bin"
idx_file = "example.idx"

# Initialize the MMapIndexedDatasetBuilder
builder = MMapIndexedDatasetBuilder(out_file=bin_file, dtype=np.int64)

# Create the data: an array with the number 1 repeated 300,005 times
data = np.ones(300005, dtype=np.int64)

# Add the data to the builder as a single document
builder.add_item(data)
builder.end_document()

# Finalize the builder to write the index file
builder.finalize(index_file=idx_file)

print("Created .bin and .idx files:")
print(f" - {bin_file}")
print(f" - {idx_file}")
