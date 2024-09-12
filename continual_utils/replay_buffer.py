import torch
import numpy as np
import os

class Buffer:
    def __init__(self, buffer_size, device="cpu", cache_size=1000):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.texts = [None] * self.buffer_size
        self.cache_size = cache_size
        self.cache = []
        self.cache_index = 0

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def add_data(self, texts: list):
        for text in texts:
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.texts[index] = text 

    def get_data(self, size: int, return_index=False):
        if not self.cache:
            self._refill_cache()

        num_avail_samples = min(len(self.cache), size)
        sampled_texts = self.cache[:num_avail_samples]
        self.cache = self.cache[num_avail_samples:]

        if not return_index:
            return sampled_texts
        else:
            return torch.arange(num_avail_samples).to(self.device), sampled_texts

    def _refill_cache(self):
        num_avail_samples = min(self.num_seen_examples, self.buffer_size)
        choice = np.random.choice(num_avail_samples, size=self.cache_size, replace=False)
        self.cache = [self.texts[i] for i in choice if self.texts[i] is not None]

    def is_empty(self) -> bool:
        return self.num_seen_examples == 0

    def empty(self) -> None:
        self.num_seen_examples = 0
        self.texts = [None] * self.buffer_size
        self.cache = []

    def save(self, filepath: str) -> None:
        buffer_data = {
            'texts': self.texts,
            'num_seen_examples': self.num_seen_examples,
            'buffer_size': self.buffer_size
        }
        torch.save(buffer_data, filepath)

    @classmethod
    def load(cls, filepath: str, device="cpu", cache_size=1000):
        buffer_data = torch.load(filepath, map_location=device)
        buffer = cls(buffer_data['buffer_size'], device, cache_size)
        buffer.texts = buffer_data['texts']
        buffer.num_seen_examples = buffer_data['num_seen_examples']
        buffer._refill_cache()
        return buffer

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    if num_seen_examples < buffer_size:
        return num_seen_examples
    rand = np.random.randint(0, num_seen_examples + 1)
    return rand if rand < buffer_size else -1

def get_all_data(self):
    """Return all the data stored in the buffer."""
    if self.num_seen_examples == 0:
        return [], [] 
    return self.texts[:self.num_seen_examples], self.task_labels[:self.num_seen_examples]

def get_data_by_index(self, indexes):
    """Get data from the buffer using specified indices."""
    sampled_texts = [self.texts[i] for i in indexes]
    sampled_task_labels = self.task_labels[indexes]
    return sampled_texts, sampled_task_labels