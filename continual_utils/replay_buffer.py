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
        if load_previous and os.path.exists(os.path.join(self.data_dir, 'metadata.json')):
            time.sleep(1)
            self.load_metadata(self.data_dir)
        else:
            self.create_files()

        self.tensor_shape = None
        self.dtype = None
        self.dtype_size = None
        self.seq_length = None 
        dummy_tensor = torch.zeros(1,2049, dtype=torch.int)
        self.prefetch_lock = Lock()
        self.file_lock = Lock()
        self.current_file_idx = 0
        self.total_samples_seen = 0
        self.add(dummy_tensor)

        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
        self.prefetch_thread.start()

    
    def create_files(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for file in self.files:
            if not os.path.exists(file):
                # Create an empty file without preallocating memory
                with open(file, 'wb') as f:
                    pass  

    def add(self, tensor_data):
        num_samples = tensor_data.shape[0]
        self.seq_length = tensor_data.shape[1]
        self.tensor_shape = tensor_data.shape
        self.dtype = tensor_data.dtype
        self.dtype_size = tensor_data.dtype.itemsize
        
        with self.file_lock:
            available_files = [idx for idx in range(self.num_files) 
                            if self.file_sizes[idx] + num_samples <= self.file_size]
            
            if available_files:
                if self.current_file_idx in available_files:
                    file_idx = self.current_file_idx
                else:
                    file_idx = available_files[0]
                    self.current_file_idx = file_idx
                    
                current_size = self.file_sizes[file_idx]
                current_file = self.files[self.current_file_idx]
                
                with open(current_file, 'r+b') as f:
                    bytes_per_sample = self.seq_length * self.dtype_size
                    f.seek(current_size * bytes_per_sample)
                    f.write(tensor_data.cpu().numpy().tobytes())
                
                current_size += num_samples
                self.file_sizes[self.current_file_idx] = current_size
            
            else:
                j = random.randint(0, self.buffer_size)
                file_idx = j // self.file_size
                position = j % self.file_size
                
                with open(self.files[file_idx], 'r+b') as f:
                    bytes_per_sample = self.seq_length * self.dtype_size
                    offset = position * bytes_per_sample
                    f.seek(offset)
                    f.write(tensor_data.cpu().numpy().tobytes())
            
            self.total_samples_seen += num_samples
            self.save_metadata(self.data_dir)

    def _prefetch_files(self):
        while True:
            if len(self.prefetch_queue) < self.prefetch_size:
                valid_file_indices = [i for i, size in enumerate(self.file_sizes) if size > 0]
                if valid_file_indices:
                    file_idx = random.choice(valid_file_indices)
                    with self.file_lock:
                        with open(self.files[file_idx], 'rb') as f:
                            num_elements = self.file_sizes[file_idx] * self.seq_length
                            buffer = f.read(num_elements * self.dtype_size)
                            
                            data = torch.frombuffer(buffer, dtype=self.dtype)
                            data = data.reshape(self.file_sizes[file_idx], self.seq_length)
                            
                            with self.prefetch_lock:
                                self.prefetch_queue.append((data, file_idx))
            else:
                time.sleep(0.1)

    def get_batch(self, buffer_proportion=0.5):
        buffer_batch_size = int(self.neox_args.eff_batch_size * buffer_proportion)
        
        with self.prefetch_lock:
            if not self.prefetch_queue:
                print("Prefetch queue is empty")
                return None

            buffer_data, file_idx = self.prefetch_queue.popleft()
            available_samples = min(len(buffer_data), buffer_batch_size)
            if available_samples > 0:
                indices = torch.randperm(len(buffer_data))[:available_samples]
                selected_data = buffer_data[indices].to(self.device)  # Shape: (buffer_batch_size x seq_length)
                return selected_data
            else:
                print("No available samples to fetch")
                return None



    def save_metadata(self, path):

        metadata = {
            # Buffer configuration
            'buffer_size': self.buffer_size,
            'file_size': self.file_size,
            'file_sizes': self.file_sizes,
            'num_files': self.num_files,
            
            # Tensor properties
            'tensor_shape':self.tensor_shape if self.tensor_shape is not None else None,  # Skip batch dimension
            'dtype': str(self.dtype) if self.dtype is not None else None,
            'dtype_size': self.dtype_size,
            
            # Statistics
            'total_samples_seen': self.total_samples_seen,
            'current_file_idx': self.current_file_idx,

        }
        
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def load_metadata(self, path):
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
                data = f.read()
                if not data.strip():
                    raise ValueError("JSON file is empty.")
                metadata = json.loads(data)
        
        # Load buffer configuration
        self.buffer_size = metadata['buffer_size']
        self.file_size = metadata['file_size']
        self.file_sizes = metadata['file_sizes']
        self.num_files = metadata['num_files']
        
        # Load tensor properties if available
        if metadata['tensor_shape'] is not None:
            self.tensor_shape = metadata["tensor_shape"]
        if metadata['dtype'] is not None:
            self.dtype = getattr(torch, metadata['dtype'].split('.')[-1])
        self.dtype_size = metadata['dtype_size']
        
        self.total_samples_seen = metadata['total_samples_seen']
        self.current_file_idx = metadata['current_file_idx']


    def __len__(self):
        return sum(self.file_sizes)
    