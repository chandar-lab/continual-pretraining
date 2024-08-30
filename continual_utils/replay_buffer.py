

from typing import List, Tuple
import numpy as np
import torch


import torch
import numpy as np
import os
from typing import List, Tuple

class Buffer:
    """
    Memory buffer for rehearsal methods in continual learning.
    Specifically designed to store text data and their corresponding labels.
    """

    def __init__(self, buffer_size, device="cpu"):
        """
        Initialize the buffer.

        Args:
            buffer_size (int): Maximum capacity of the buffer.
            device (str, optional): Device to store the buffer tensors. Defaults to "cpu".
        """
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['texts', 'task_labels']  # Attributes stored in the buffer

    def to(self, device):
        """Move the buffer tensors to the specified device."""
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        """Return the current number of elements in the buffer."""
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self) -> None:
        """Initialize the tensors for storing text and task_label data."""
        self.texts = [None] * self.buffer_size  # List to hold text strings
        self.task_labels = [None] * self.buffer_size  # List to hold task labels

    def add_data(self, texts: List[str], task_labels: List[str]):
        """
        Add data to the buffer using reservoir sampling.

        Args:
            texts (List[str]): List of text strings.
            task_labels (List[str]): List of corresponding task labels
        """
        if not hasattr(self, 'texts'):
            self.init_tensors()

        for text, task_label in zip(texts, task_labels):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.texts[index] = text 
                self.task_labels[index] = task_label

    def get_data(self, size: int, return_index=False) -> Tuple:
        """
        Sample a batch of data from the buffer.

        Args:
            size (int): Batch size.
            return_index (bool, optional): Whether to return the indices of the sampled data. Defaults to False.

        Returns:
            Tuple: A tuple containing the sampled texts and task_labels. If return_index is True, the indices are included as the first element.
        """
        num_avail_samples = min(self.num_seen_examples, self.buffer_size)
        if size > num_avail_samples:
            size = num_avail_samples

        choice = np.random.choice(num_avail_samples, size=size, replace=False)
        sampled_texts = [self.texts[i] for i in choice]
        sampled_task_labels = [self.task_labels[i] for i in choice]

        if not return_index:
            return sampled_texts, sampled_task_labels 
        else:
            return torch.tensor(choice).to(self.device), sampled_texts, sampled_task_labels

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return self.num_seen_examples == 0

    def empty(self) -> None:
        """Empty the buffer."""
        self.num_seen_examples = 0
        del self.texts, self.task_labels

    def save_buffer(self, filepath: str) -> None:
        """
        Save the buffer to a file.

        Args:
            filepath (str): The path to the file where the buffer will be saved.
        """
        buffer_data = {
            'num_seen_examples': self.num_seen_examples,
            'texts': self.texts,
            'task_labels': self.task_labels
        }
        torch.save(buffer_data, filepath)

    def load_buffer(self, filepath: str) -> None:
        """
        Load the buffer from a file.

        Args:
            filepath (str): The path to the file from which the buffer will be loaded.
        """
        if os.path.isfile(filepath):
            buffer_data = torch.load(filepath, map_location=self.device)
            self.num_seen_examples = buffer_data['num_seen_examples']
            self.texts = buffer_data['texts']
            self.task_labels = buffer_data['task_labels']
        else:
            raise FileNotFoundError(f"No such file: '{filepath}'")


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """Reservoir sampling strategy for adding data to the buffer."""
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

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


    
