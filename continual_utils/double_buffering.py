import os
import json
import time
import torch
import random
from collections import deque
from threading import Lock, Thread

class ReplayBuffer:
    def __init__(self, neox_args, device=torch.device('cuda'), prefetch_size=5):
        self.neox_args = neox_args
        self.buffer_size = neox_args.buffer_size
        self.file_size = neox_args.file_size
        self.data_dir = neox_args.buffer_dir
        self.device = device
        self.prefetch_size = prefetch_size
        self.number_of_buffers = self.buffer_size // self.file_size
        self.files = [f"{self.data_dir}/buffer_{i}.pt" for i in range(self.number_of_buffers)]
        self.file_sizes = [0] * self.number_of_buffers
        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.prefetch_lock = Lock()
        self.file_lock = Lock()
        
        # Double buffering setup
        self.active_buffer = deque(maxlen=self.file_size)
        self.secondary_buffer = deque(maxlen=self.file_size)
        self.current_buffer = self.active_buffer
        self.secondary_write_thread = None

        self.create_files()
        self.prefetch_thread = Thread(target=self._prefetch_files, daemon=True)
        self.prefetch_thread.start()

        self.reservoir = []  
        self.reservoir_size = 0  

    def create_files(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for file in self.files:
            if not os.path.exists(file):
                with open(file, 'wb') as f:
                    pass

    def add(self, tensor_data):
        num_samples = tensor_data.shape[0]
        
        for i in range(num_samples):
            if len(self.current_buffer) < self.file_size:
                self.current_buffer.append(tensor_data[i].cpu())
            else:
                # Buffer is full; initiate write to disk for the current buffer
                if self.secondary_write_thread is None or not self.secondary_write_thread.is_alive():
                    self.secondary_write_thread = Thread(target=self._write_buffer_to_disk, args=(self.current_buffer,))
                    self.secondary_write_thread.start()
                    
                # Swap buffers
                self.current_buffer = self.secondary_buffer if self.current_buffer == self.active_buffer else self.active_buffer
                self.current_buffer.clear()  # Clear the new buffer before use
                self.current_buffer.append(tensor_data[i].cpu())

    def _write_buffer_to_disk(self, buffer):
        file_idx = random.choice([i for i in range(len(self.file_sizes)) if self.file_sizes[i] < self.file_size])
        file_path = self.files[file_idx]
        
        with self.file_lock:
            with open(file_path, 'ab') as f:
                for tensor in buffer:
                    f.write(tensor.numpy().tobytes())
                    self.file_sizes[file_idx] += 1
                    if self.file_sizes[file_idx] >= self.file_size:
                        break

    def _update_reservoir(self, tensor_data):
        num_samples = tensor_data.shape[0]
        for i in range(num_samples):
            if self.reservoir_size < self.prefetch_size:
                self.reservoir.append(tensor_data[i].cpu())
                self.reservoir_size += 1
            else:
                replace_idx = random.randint(0, self.reservoir_size - 1)
                if replace_idx < self.prefetch_size:
                    self.reservoir[replace_idx] = tensor_data[i].cpu()

    def _prefetch_files(self):
        while True:
            target_buffer = self.secondary_buffer
