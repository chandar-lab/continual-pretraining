import os
import random
import numpy as np
import torch
from collections import deque
from threading import Thread, Lock
import time

class ReplayBuffer:
    def __init__(self, neox_args, device = torch.device('cuda'), prefetch_size=100):
        self.neox_args = neox_args
        self.buffer_size = self.neox_args.buffer_size 
        self.file_size = self.neox_args.file_size
        self.data_dir = self.neox_args.buffer_dir
        self.device = device
        self.prefetch_size = prefetch_size
        
        self.num_files = (self.buffer_size + self.file_size - 1) // self.file_size
        self.current_size = 0
        self.total_samples_seen = 0
        
        self.files = [f"{self.data_dir}/buffer_{i}.npy" for i in range(self.num_files)]
        self.create_files()
        
        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.prefetch_lock = Lock()
        self.prefetch_thread = Thread(target=self._prefetch_file, daemon=True)
        self.prefetch_thread.start()

    def create_files(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for file in self.files:
            if not os.path.exists(file):
                np.save(file, np.zeros((self.file_size,), dtype=np.int64))

    def add(self, sample):
        if self.current_size < self.buffer_size:
            file_idx = self.current_size // self.file_size
            sample_idx = self.current_size % self.file_size
            self._write_to_file(file_idx, sample_idx, sample)
            self.current_size += 1
        else:
            if random.random() < self.buffer_size / (self.total_samples_seen + 1):
                replace_idx = random.randint(0, self.buffer_size - 1)
                file_idx = replace_idx // self.file_size
                sample_idx = replace_idx % self.file_size
                self._write_to_file(file_idx, sample_idx, sample)
        
        self.total_samples_seen += 1

    def _write_to_file(self, file_idx, sample_idx, sample):
        file = self.files[file_idx]
        data = np.load(file)
        data[sample_idx] = sample
        np.save(file, data)

    def _prefetch_file(self):
        while True:
            if len(self.prefetch_queue) < self.prefetch_size:
                file_idx = random.randint(0, self.num_files - 1)
                file = self.files[file_idx]
                data = np.load(file)
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
                file_idx = random.randint(0, self.num_files - 1)
                file = self.files[file_idx]
                buffer_data = np.load(file)

        if buffer_size > len(buffer_data):
            buffer_size = len(buffer_data)

        indices = np.random.choice(len(buffer_data), buffer_size, replace=False)
        buffer_samples = buffer_data[indices]

        return torch.tensor(buffer_samples, dtype=torch.long, device=self.device)

    def __len__(self):
        return self.current_size





# import torch
# import numpy as np
# import os
# import threading
# import queue

# class ReplayBuffer:
#     def __init__(self, buffer_size: int, file_size: int, data_dir: str, device: str = "cpu", prefetch_size: int = 10):
#         self.buffer_size = buffer_size
#         self.file_size = file_size
#         self.data_dir = data_dir
#         self.device = device
#         self.num_seen_examples = 0
        
#         self.num_files = (buffer_size + file_size - 1) // file_size
#         self.files = [None] * self.num_files
        
#         # Ensure data directory exists
#         os.makedirs(self.data_dir, exist_ok=True)
        
#         # Prefetch queue
#         self.prefetch_size = prefetch_size
#         self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
#         self.prefetch_thread = threading.Thread(target=self._prefetch_files, daemon=True)
#         self.prefetch_thread.start()

#     def add_data(self, tokens: torch.Tensor):
#         tokens = tokens.to("cpu").flatten()
#         for token in tokens:
#             if self.num_seen_examples < self.buffer_size:
#                 index = self.num_seen_examples
#             else:
#                 index = np.random.randint(0, self.num_seen_examples + 1)
#                 if index >= self.buffer_size:
#                     self.num_seen_examples += 1
#                     continue
            
#             file_idx = index // self.file_size
#             local_idx = index % self.file_size
            
#             if self.files[file_idx] is None:
#                 self.files[file_idx] = torch.zeros(self.file_size, dtype=torch.int64)
            
#             self.files[file_idx][local_idx] = token
#             self._save_file(file_idx)
            
#             self.num_seen_examples += 1

#     def get_data(self, size: int) -> torch.Tensor:
#         if self.num_seen_examples == 0:
#             return torch.tensor([], dtype=torch.int64, device=self.device)
        
#         indices = np.random.choice(min(self.num_seen_examples, self.buffer_size), size, replace=True)
#         result = torch.zeros(size, dtype=torch.int64, device=self.device)
        
#         for i, idx in enumerate(indices):
#             file_idx = idx // self.file_size
#             local_idx = idx % self.file_size
            
#             file_data = self._get_file(file_idx)
#             result[i] = file_data[local_idx].to(self.device)
        
#         return result

#     def _save_file(self, file_idx: int):
#         file_path = os.path.join(self.data_dir, f"buffer_{file_idx}.pt")
#         torch.save(self.files[file_idx], file_path)

#     def _get_file(self, file_idx: int) -> torch.Tensor:
#         if self.files[file_idx] is None:
#             try:
#                 file_data = self.prefetch_queue.get_nowait()
#                 self.files[file_idx] = file_data
#             except queue.Empty:
#                 file_path = os.path.join(self.data_dir, f"buffer_{file_idx}.pt")
#                 self.files[file_idx] = torch.load(file_path, map_location="cpu")
#         return self.files[file_idx]

#     def _prefetch_files(self):
#         while True:
#             file_indices = np.random.choice(self.num_files, self.prefetch_size, replace=False)
#             for file_idx in file_indices:
#                 if self.files[file_idx] is None:
#                     file_path = os.path.join(self.data_dir, f"buffer_{file_idx}.pt")
#                     file_data = torch.load(file_path, map_location="cpu")
#                     self.prefetch_queue.put(file_data)

#     def __len__(self):
#         return min(self.num_seen_examples, self.buffer_size)

#     def save_state(self, path: str):
#         state = {
#             'num_seen_examples': self.num_seen_examples,
#             'buffer_size': self.buffer_size,
#             'file_size': self.file_size,
#             'data_dir': self.data_dir,
#         }
#         torch.save(state, path)

#     @classmethod
#     def load_state(cls, path: str, device: str = "cpu", prefetch_size: int = 10):
#         state = torch.load(path)
#         buffer = cls(
#             buffer_size=state['buffer_size'],
#             file_size=state['file_size'],
#             data_dir=state['data_dir'],
#             device=device,
#             prefetch_size=prefetch_size
#         )
#         buffer.num_seen_examples = state['num_seen_examples']
#         return buffer

# Example usage
# if __name__ == "__main__":
#     buffer = ReplayBuffer(
#         buffer_size=1000000,
#         file_size=10000,
#         data_dir="./buffer_data",
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )

#     # Add some data
#     for _ in range(10):
#         buffer.add_data(torch.randint(0, 1000, (1000,)))

#     # Get some data
#     data = buffer.get_data(500)
#     print(f"Retrieved {len(data)} tokens from buffer")

#     # Save and load buffer state
#     buffer.save_state("buffer_state.pt")
#     loaded_buffer = ReplayBuffer.load_state("buffer_state.pt")
#     print(f"Loaded buffer with {len(loaded_buffer)} tokens")

