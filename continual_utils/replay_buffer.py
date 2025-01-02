import os
import json
import time
import torch
import torch.distributed as dist
from collections import deque
from threading import Lock, Thread
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, neox_args, device=torch.device('cuda'), prefetch_size=50, load_previous=False):
        self.neox_args = neox_args
        self.buffer_size = neox_args.buffer_size  # Total buffer size in tokens
        self.file_size = neox_args.file_size      # Size of each file in tokens
        self.data_dir = neox_args.buffer_dir       # Directory to store buffer files
        self.device = device
        self.prefetch_size = prefetch_size
        self.num_files = self.buffer_size // self.file_size
        self.files = [f"{self.data_dir}/buffer_{i}.pt" for i in range(self.buffer_size // self.file_size)]
        self.file_sizes = [0] * len(self.files)  
        self.tensor_shape = None
        self.dtype = None
        self.dtype_size = None
        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.prefetch_lock = Lock()
        self.file_lock = Lock()
        if load_previous and os.path.exists(os.path.join(self.data_dir, 'metadata.json')):
            self.load_previous_state()
        else:
            self.create_files()
        self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
        self.prefetch_thread.start()
        self.current_file_idx = 0
        self.total_samples_seen = 0
        
    def load_previous_state(self):
        """Load the previous state of the buffer including metadata and tensor properties"""
        # Load metadata first
        self.load_metadata(self.data_dir)
        
        # Find first non-empty file to get tensor properties
        for idx, size in enumerate(self.file_sizes):
            if size > 0:
                with open(self.files[idx], 'rb') as f:
                    # Read first sample to get tensor properties
                    sample_data = f.read(size * self.dtype_size)
                    tensor = torch.frombuffer(sample_data, dtype=self.dtype)
                    self.tensor_shape = tensor.shape
                    break
        
        if self.tensor_shape is None:
            # If no files had data, create new files
            self.create_files()
            return
            
        # Verify all files exist, create any missing ones
        for file in self.files:
            if not os.path.exists(file):
                with open(file, 'wb') as f:
                    pass
    
    def create_files(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for file in self.files:
            if not os.path.exists(file):
                # Create an empty file without preallocating memory
                with open(file, 'wb') as f:
                    pass  

    def add(self, tensor_data):
        num_samples = tensor_data.shape[0]
        self.tensor_shape = tensor_data.shape
        self.dtype = tensor_data.dtype
        self.dtype_size = tensor_data.dtype.itemsize
        
        with self.file_lock:
            # Get list of files that have space for this batch
            available_files = [idx for idx in range(self.num_files) 
                            if self.file_sizes[idx] + num_samples <= self.file_size]
            
            if available_files:
                # If current file has space, use it; otherwise use the first available file
                if self.current_file_idx in available_files:
                    file_idx = self.current_file_idx
                else:
                    file_idx = available_files[0]
                    self.current_file_idx = file_idx
                    
                current_size = self.file_sizes[file_idx]
                current_file = self.files[self.current_file_idx]
                
                # Write tensor_data to file
                with open(current_file, 'r+b') as f:
                    bytes_per_sample = tensor_data.shape[1] * self.dtype_size
                    f.seek(current_size * bytes_per_sample)
                    f.write(tensor_data.cpu().numpy().tobytes())
                
                current_size += num_samples
                self.file_sizes[self.current_file_idx] = current_size
            
            else:
                # All files are full, implement reservoir sampling using buffer size
                j = random.randint(0, self.buffer_size)
                # Calculate which file and position to replace
                file_idx = j // self.file_size
                position = j % self.file_size
                
                self.current_file_idx = file_idx
                # Write at the chosen position
                with open(self.files[file_idx], 'r+b') as f:
                    bytes_per_sample = tensor_data.shape[1] * self.dtype_size
                    offset = position * bytes_per_sample
                    f.seek(offset)
                    f.write(tensor_data.cpu().numpy().tobytes())
            
            self.total_samples_seen += num_samples
            self.save_metadata(self.data_dir)


    def _prefetch_files(self):
        while True:
            if len(self.prefetch_queue) < self.prefetch_size:
                # Only prefetch files that have data
                valid_file_indices = [i for i, size in enumerate(self.file_sizes) if size > 0]
                if valid_file_indices:
                    file_idx = random.choice(valid_file_indices)
                    with self.file_lock:
                        # Read the bytes data and convert back to tensor
                        with open(self.files[file_idx], 'rb') as f:
                            # Read only the filled portion of the file
                            num_elements = self.file_sizes[file_idx] * self.tensor_shape[1]
                            buffer = f.read(num_elements * self.dtype_size)
                            
                            # Convert bytes directly to tensor
                            data = torch.frombuffer(buffer, dtype=self.dtype)
                            data = data.reshape(self.file_sizes[file_idx], self.tensor_shape[1])
                            
                            with self.prefetch_lock:
                                self.prefetch_queue.append((data, file_idx))
            else:
                time.sleep(0.1)

    def get_batch(self, buffer_proportion=0.5):
        # self.neox_args.eff_batch_size=neox_args.batch_size
        buffer_batch_size = int(self.neox_args.eff_batch_size * buffer_proportion)
        
        with self.prefetch_lock:
            if not self.prefetch_queue:
                print("Prefetch queue is empty")
                return None

            buffer_data, file_idx = self.prefetch_queue.popleft()
            available_samples = min(len(buffer_data), buffer_batch_size)  # Ensure it does not exceed buffer_data length

            if available_samples > 0:
                indices = torch.randperm(len(buffer_data))[:available_samples]  # Use len(buffer_data) to ensure validity
                selected_data = buffer_data[indices].to(self.device)
                return selected_data
            else:
                print("No available samples to fetch")
                return None



    def save_metadata(self, path):
        # print(self.tensor_shape[1:])

        metadata = {
            # Buffer configuration
            'buffer_size': self.buffer_size,
            'file_size': self.file_size,
            'file_sizes': self.file_sizes,
            'num_files': self.num_files,
            
            # Tensor properties
            'tensor_shape': list(self.tensor_shape[1:]) if self.tensor_shape is not None else None,  # Skip batch dimension
            'dtype': str(self.dtype) if self.dtype is not None else None,
            'dtype_size': self.dtype_size,
            
            # Statistics
            'total_samples_seen': self.total_samples_seen,
            'current_file_idx': self.current_file_idx,

        }
        
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, path):
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Load buffer configuration
        self.buffer_size = metadata['buffer_size']
        self.file_size = metadata['file_size']
        self.file_sizes = metadata['file_sizes']
        self.num_files = metadata['num_files']
        
        # Load tensor properties if available
        if metadata['tensor_shape'] is not None:
            self.tensor_shape = (-1, *metadata['tensor_shape'])  # Restore batch dimension
        if metadata['dtype'] is not None:
            self.dtype = getattr(torch, metadata['dtype'].split('.')[-1])
        self.dtype_size = metadata['dtype_size']
        
        # Load statistics
        self.total_samples_seen = metadata['total_samples_seen']
        self.current_file_idx = metadata['current_file_idx']


    def __len__(self):
        return sum(self.file_sizes)
    