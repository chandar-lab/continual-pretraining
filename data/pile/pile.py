import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        # Assuming data is stored as 32-bit integers in little-endian format
        dtype = np.dtype(np.int32).newbyteorder('<')
        data = np.fromfile(f, dtype=dtype)
    return data

# Usage
idx_data = read_idx('train/pile_train.idx')
print("Class labels:", idx_data)