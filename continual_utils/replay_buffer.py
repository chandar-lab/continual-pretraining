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
    def __init__(self, neox_args, device=torch.device('cuda'), prefetch_size=5):
        self.neox_args = neox_args
        self.buffer_size = neox_args.buffer_size  # Total buffer size in tokens
        self.file_size = neox_args.file_size      # Size of each file in tokens
        self.data_dir = neox_args.buffer_dir       # Directory to store buffer files
        self.device = device
        self.prefetch_size = prefetch_size
        self.num_files = self.buffer_size // self.file_size
        self.files = [f"{self.data_dir}/buffer_{i}.pt" for i in range(self.buffer_size // self.file_size)]
        self.file_sizes = [0] * len(self.files)  
        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.prefetch_lock = Lock()
        self.file_lock = Lock()
        self.create_files()
        self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
        self.prefetch_thread.start()
        self.current_file_idx = 0
        self.total_samples_seen = 0

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
        """
        Get a batch of samples from the buffer.
        buffer_proportion: proportion of samples to get from the buffer (0.0 to 1.0)
        """
        buffer_batch_size = int(self.neox_args.eff_batch_size * buffer_proportion)
        
        with self.prefetch_lock:
            if not self.prefetch_queue:
                return None

            # Get data from prefetch queue
            buffer_data, file_idx = self.prefetch_queue.popleft()
            
            # Calculate how many samples we can actually get from this file
            available_samples = min(self.file_sizes[file_idx], buffer_batch_size)
            
            # Randomly sample from the available samples
            if available_samples > 0:
                indices = torch.randperm(available_samples)[:buffer_batch_size]
                return buffer_data[indices].to(self.device)
            else:
                return None

    def save_metadata(self, path):
        metadata = {
            'file_sizes': self.file_sizes,
            'buffer_size': self.buffer_size
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def load_metadata(self, path):
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        self.file_sizes = metadata['file_sizes']
        self.buffer_size = metadata['buffer_size']

    def __len__(self):
        return sum(self.file_sizes)
    
# class ReplayBuffer:
#     def __init__(self, neox_args, device=torch.device('cuda'), prefetch_size=5):
#         self.neox_args = neox_args
#         self.buffer_size = neox_args.buffer_size  # Total buffer size in tokens
#         self.file_size = neox_args.file_size      # Size of each file in tokens
#         self.data_dir = neox_args.buffer_dir       # Directory to store buffer files
#         self.device = device
#         self.prefetch_size = prefetch_size

#         # Initialize rank and world size for distributed training
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.world_size = dist.get_world_size() if dist.is_initialized() else 1

#         # Create file paths for each rank
#         self.files = [f"{self.data_dir}/buffer_{i}.pt" for i in range(self.world_size)]

#         # Initialize buffer state
#         self.current_size = 0
#         self.total_samples_seen = 0
#         self.reservoir = []  # For reservoir sampling

#         # Prefetching setup
#         self.prefetch_queue = deque(maxlen=prefetch_size)
#         self.prefetch_lock = Lock()
#         self.file_lock = Lock()
#         self.create_files()
#         self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
#         self.prefetch_thread.start()



#     def create_files(self):
#         os.makedirs(self.data_dir, exist_ok=True)
#         for file in self.files:
#             if not os.path.exists(file):
#                 empty_tensor = torch.zeros(self.file_size, 2049, dtype=torch.int64)
#                 with self.file_lock:
#                     torch.save(empty_tensor, file)

#     def add(self, tensor_data):
#         # Implement reservoir sampling
#         if self.current_size < self.buffer_size:
#             self._add_to_reservoir(tensor_data)
#             self.current_size += 1
#         self.total_samples_seen += 1

#     def _add_to_reservoir(self, sample):
#         if len(self.reservoir) < self.buffer_size:
#             self.reservoir.append(sample.cpu())
#         else:
#             replace_idx = random.randint(0, self.total_samples_seen - 1)
#             if replace_idx < self.buffer_size:
#                 self.reservoir[replace_idx] = sample.cpu()

#     def _write_to_file(self, file_idx):
#         file = self.files[file_idx]
#         with self.file_lock:
#             data = torch.zeros(self.file_size, 2049, dtype=torch.int64)
#             for idx, sample in enumerate(self.reservoir):
#                 if idx < self.file_size:
#                     data[idx] = sample
#             torch.save(data, file)

#     def _prefetch_files(self):
#         while True:
#             if len(self.prefetch_queue) < self.prefetch_size:
#                 file_idx = random.randint(0, self.world_size - 1)
#                 file = self.files[file_idx]
#                 with self.file_lock:
#                     data = torch.load(file, map_location='cpu')
#                     with self.prefetch_lock:
#                         self.prefetch_queue.append(data)
#             else:
#                 time.sleep(0.1)

#     def get_batch(self, buffer_proportion=0.5):
#         buffer_size = int(self.neox_args.eff_batch_size * buffer_proportion)
#         with self.prefetch_lock:
#             if self.prefetch_queue:
#                 buffer_data = self.prefetch_queue.popleft()
#             else:
#                 file_idx = random.randint(0, self.world_size - 1)
#                 file = self.files[file_idx]
#                 with self.file_lock:
#                     buffer_data = torch.load(file, map_location='cpu')

#         if len(buffer_data) < buffer_size:
#             return None

#         indices = torch.randperm(len(buffer_data))[:buffer_size]
#         buffer_samples = buffer_data[indices]
#         return buffer_samples.to(self.device)
# class ReplayBuffer:
#     def __init__(self, neox_args, device=torch.device('cuda'), prefetch_size=5):
#         self.neox_args = neox_args
#         self.buffer_size = neox_args.buffer_size
#         self.file_size = neox_args.file_size
#         self.data_dir = neox_args.buffer_dir
#         self.device = device
#         self.prefetch_size = prefetch_size
#         # self.init_distributed_mode()
#         # dist.init_process_group(backend='nccl')
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         # self.world_size = dist.get_world_size() if dist.is_initialized() else 1
#         self.world_size = neox_args.buffer_size/neox_args.file_size

#         self.files = [f"{self.data_dir}/buffer_{i}.pt" for i in range(int(self.world_size))]
#         self.file_sizes = [0] * int(self.world_size) # Track the actual data size in each file

#         self.current_size = 0
#         self.total_samples_seen = 0
#         self.reservoir = []

#         self.prefetch_queue = deque(maxlen=prefetch_size)
#         self.prefetch_lock = Lock()
#         self.file_lock = Lock()
#         self.create_files()
#         self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
#         self.prefetch_thread.start()

#     def init_distributed_mode(self):
#         dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
#         world_size = dist.get_world_size()
#         rank = dist.get_rank()
#         print(f"Initialized process group with {world_size} processes. Current rank: {rank}")
    
#     def create_files(self):
#         os.makedirs(self.data_dir, exist_ok=True)
#         for file in self.files:
#             if not os.path.exists(file):
#                 empty_tensor = torch.zeros(self.file_size, 2049, dtype=torch.int32)  # Adjust dtype if needed
#                 with self.file_lock:
#                     torch.save(empty_tensor, file)


#     def add(self, tensor_data):
#         num_samples = tensor_data.shape[0] # Account for batch size

#         if self.current_size + num_samples <= self.buffer_size:  # Check if there is enough space
#              for i in range(num_samples):
#                 self._add_to_reservoir(tensor_data[i])
#              self.current_size += num_samples #Increment by number of samples added
#         else:
#             # handle the case where there's not enough space
#             remaining_space = self.buffer_size - self.current_size
#             if remaining_space > 0:
#                 for i in range(remaining_space):
#                     self._add_to_reservoir(tensor_data[i])
#                 self.current_size += remaining_space

#         self.total_samples_seen += num_samples

#     def _add_to_reservoir(self, sample): #This function deals with individual samples now
#         if len(self.reservoir) < self.buffer_size:
#             self.reservoir.append(sample.cpu())
#         else:
#             replace_idx = random.randint(0, self.total_samples_seen - 1)
#             if replace_idx < self.buffer_size:
#                 self.reservoir[replace_idx] = sample.cpu()


#     def _write_to_file(self, file_idx):
#         file = self.files[file_idx]
#         with self.file_lock:
#             data = torch.zeros(self.file_size, 2049, dtype=torch.int32) #Adjust dtype if needed
#             num_to_write = min(len(self.reservoir), self.file_size)
#             for idx, sample in enumerate(self.reservoir[:num_to_write]): # Write up to file_size elements
#                 data[idx] = sample
#             torch.save(data, file)
#             self.file_sizes[file_idx] = num_to_write #update file size


#     def _prefetch_files(self):
#         while True:
#             if len(self.prefetch_queue) < self.prefetch_size:
#                 # Prioritize files which are full or closest to full, then fallback to random.
#                 valid_file_indices = [i for i, size in enumerate(self.file_sizes) if size > 0]
#                 if valid_file_indices:
#                     file_idx = random.choices(valid_file_indices, weights=self.file_sizes, k=1)[0]
#                 else:  # If no files are filled yet, choose randomly
#                     file_idx = random.randint(0, int(self.world_size) - 1)


#                 file = self.files[file_idx]
#                 with self.file_lock:  # Ensure atomicity when reading file sizes
#                      data = torch.load(file, map_location='cpu')

#                      with self.prefetch_lock:
#                         self.prefetch_queue.append((data, file_idx)) # Keep track of file idx
#             else:
#                 time.sleep(0.1)



#     def get_batch(self, buffer_proportion=0.5):
#         buffer_size = int(self.neox_args.eff_batch_size * buffer_proportion)

#         with self.prefetch_lock:
#             if self.prefetch_queue:
#                 buffer_data, file_idx = self.prefetch_queue.popleft() #Retrieve file index
#                 actual_size = self.file_sizes[file_idx]
#                 if actual_size < buffer_size:
#                     indices = torch.randperm(actual_size)[:buffer_size] # Sample within actual data range
#                 else:
#                     indices = torch.randperm(actual_size)[:buffer_size]

#                 buffer_samples = buffer_data[indices]
#                 return buffer_samples.to(self.device)
#             else:
#                 return None # Handle empty queue
            
#     def save_buffer(self, path):
#         metadata = {
#             'current_size': self.current_size,
#             'total_samples_seen': self.total_samples_seen,
#             'file_size': self.file_size,
#             'buffer_size': self.buffer_size
#         }
#         with open(os.path.join(path, f'metadata_{self.rank}.json'), 'w') as f:
#             json.dump(metadata, f)

#     def load_buffer(self, path):
#         with open(os.path.join(path, f'metadata_{self.rank}.json'), 'r') as f:
#             metadata = json.load(f)
#         self.current_size = metadata['current_size']
#         self.total_samples_seen = metadata['total_samples_seen']
#         self.file_size = metadata['file_size']
#         self.buffer_size = metadata['buffer_size']
#         with self.prefetch_lock:
#             self.prefetch_queue.clear()

#     def __len__(self):
#         return self.current_size

# import os
# import torch
# import torch.distributed as dist
# from collections import deque
# from threading import Lock, Thread
# import time
# import json
# class ReplayBuffer:
#     def __init__(self, neox_args, device=torch.device('cuda'), prefetch_size=5):
#         self.neox_args = neox_args
#         self.buffer_size = self.neox_args.buffer_size
#         self.file_size = self.neox_args.file_size
#         self.data_dir = self.neox_args.buffer_dir
#         self.device = device
#         self.prefetch_size = prefetch_size

#         # Use rank-based file naming to avoid collisions
#         self.rank = dist.get_rank()  # Get the rank of the current node
#         self.files = [f"{self.data_dir}/buffer_{self.rank}.pt"]
#         self.create_files()

#         self.prefetch_queue = deque(maxlen=prefetch_size)
#         self.prefetch_lock = Lock()
#         self.file_lock = Lock()
#         self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
#         self.prefetch_thread.start()

#     def create_files(self):
#         os.makedirs(self.data_dir, exist_ok=True)
#         for file in self.files:
#             if not os.path.exists(file):
#                 empty_tensor = torch.zeros(self.file_size, 2049, dtype=torch.int64)
#                 with self.file_lock:
#                     torch.save(empty_tensor, file)

#     def add(self, tensor_data):
#         if self.current_size < self.buffer_size:
#             file_idx = 0
#             sample_idx = self.current_size % self.file_size
#             self._write_to_file(file_idx, sample_idx, tensor_data)
#             self.current_size += 1
#         self.total_samples_seen += 1

#     def _write_to_file(self, file_idx, sample_idx, sample):
#         file = self.files[file_idx]
#         with self.file_lock:
#             data = torch.load(file)
#             data[sample_idx] = sample.cpu()
#             torch.save(data, file)

#     def _prefetch_files(self):
#         while True:
#             if len(self.prefetch_queue) < self.prefetch_size:
#                 file = self.files[0]
#                 with self.file_lock:
#                     data = torch.load(file)
#                     with self.prefetch_lock:
#                         self.prefetch_queue.append(data)
#             else:
#                 time.sleep(0.1)

#     def get_batch(self, buffer_proportion=0.5):
#         buffer_size = int(self.neox_args.eff_batch_size * buffer_proportion)
#         with self.prefetch_lock:
#             if self.prefetch_queue:
#                 buffer_data = self.prefetch_queue.popleft()
#             else:
#                 file = self.files[0]
#                 with self.file_lock:
#                     buffer_data = torch.load(file)

#         indices = torch.randperm(len(buffer_data))[:buffer_size]
#         buffer_samples = buffer_data[indices]
#         return buffer_samples.to(self.device)

#     def save_buffer(self, path):
#         metadata = {
#             'current_size': self.current_size,
#             'total_samples_seen': self.total_samples_seen,
#             'file_size': self.file_size,
#             'buffer_size': self.buffer_size
#         }
#         with open(os.path.join(path, f'metadata_{self.rank}.json'), 'w') as f:
#             json.dump(metadata, f)

#     def load_buffer(self, path):
#         with open(os.path.join(path, f'metadata_{self.rank}.json'), 'r') as f:
#             metadata = json.load(f)
#         self.current_size = metadata['current_size']
#         self.total_samples_seen = metadata['total_samples_seen']
#         self.file_size = metadata['file_size']
#         self.buffer_size = metadata['buffer_size']
#         with self.prefetch_lock:
#             self.prefetch_queue.clear()