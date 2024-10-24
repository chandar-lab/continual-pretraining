import os
import json
import time
import torch
import torch.distributed as dist
from collections import deque
from threading import Lock, Thread
import random

class ReplayBuffer:
    def __init__(self, neox_args, device=torch.device('cuda'), prefetch_size=5):
        self.neox_args = neox_args
        self.buffer_size = neox_args.buffer_size  # Total buffer size in tokens
        self.file_size = neox_args.file_size      # Size of each file in tokens
        self.data_dir = neox_args.buffer_dir       # Directory to store buffer files
        self.device = device
        self.prefetch_size = prefetch_size

        # Initialize rank and world size for distributed training
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Create file paths for each rank
        self.files = [f"{self.data_dir}/buffer_{i}.pt" for i in range(self.world_size)]

        # Initialize buffer state
        self.current_size = 0
        self.total_samples_seen = 0
        self.reservoir = []  # For reservoir sampling

        # Prefetching setup
        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.prefetch_lock = Lock()
        self.file_lock = Lock()
        self.create_files()
        self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
        self.prefetch_thread.start()



    def create_files(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for file in self.files:
            if not os.path.exists(file):
                empty_tensor = torch.zeros(self.file_size, 2049, dtype=torch.int64)
                with self.file_lock:
                    torch.save(empty_tensor, file)

    def add(self, tensor_data):
        # Implement reservoir sampling
        if self.current_size < self.buffer_size:
            self._add_to_reservoir(tensor_data)
            self.current_size += 1
        self.total_samples_seen += 1

    def _add_to_reservoir(self, sample):
        if len(self.reservoir) < self.buffer_size:
            self.reservoir.append(sample.cpu())
        else:
            replace_idx = random.randint(0, self.total_samples_seen - 1)
            if replace_idx < self.buffer_size:
                self.reservoir[replace_idx] = sample.cpu()

    def _write_to_file(self, file_idx):
        file = self.files[file_idx]
        with self.file_lock:
            data = torch.zeros(self.file_size, 2049, dtype=torch.int64)
            for idx, sample in enumerate(self.reservoir):
                if idx < self.file_size:
                    data[idx] = sample
            torch.save(data, file)

    def _prefetch_files(self):
        while True:
            if len(self.prefetch_queue) < self.prefetch_size:
                file_idx = random.randint(0, self.world_size - 1)
                file = self.files[file_idx]
                with self.file_lock:
                    data = torch.load(file, map_location='cpu')
                    with self.prefetch_lock:
                        self.prefetch_queue.append(data)
            else:
                time.sleep(0.1)

    def get_batch(self, buffer_proportion=0.5):
        buffer_size = int(self.neox_args.eff_batch_size * buffer_proportion)
        with self.prefetch_lock:
            if self.prefetch_queue:
                buffer_data = self.prefetch_queue.popleft()
            else:
                file_idx = random.randint(0, self.world_size - 1)
                file = self.files[file_idx]
                with self.file_lock:
                    buffer_data = torch.load(file, map_location='cpu')

        if len(buffer_data) < buffer_size:
            return None

        indices = torch.randperm(len(buffer_data))[:buffer_size]
        buffer_samples = buffer_data[indices]
        return buffer_samples.to(self.device)

    def save_buffer(self, path):
        metadata = {
            'current_size': self.current_size,
            'total_samples_seen': self.total_samples_seen,
            'file_size': self.file_size,
            'buffer_size': self.buffer_size
        }
        with open(os.path.join(path, f'metadata_{self.rank}.json'), 'w') as f:
            json.dump(metadata, f)

    def load_buffer(self, path):
        with open(os.path.join(path, f'metadata_{self.rank}.json'), 'r') as f:
            metadata = json.load(f)
        self.current_size = metadata['current_size']
        self.total_samples_seen = metadata['total_samples_seen']
        self.file_size = metadata['file_size']
        self.buffer_size = metadata['buffer_size']
        with self.prefetch_lock:
            self.prefetch_queue.clear()

    def __len__(self):
        return self.current_size

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