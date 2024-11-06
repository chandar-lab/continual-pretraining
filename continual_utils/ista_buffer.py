from typing import List, Tuple
import numpy as np
import torch
import os
from threading import Thread, Lock

class Buffer:
    """
    Memory buffer for rehearsal methods in continual learning, with double buffering for efficient saving.
    """

    def __init__(self, buffer_size, device="cpu"):
        """
        Initialize the buffer with double buffering.

        Args:
            buffer_size (int): Maximum capacity of each buffer.
            device (str, optional): Device to store the buffer tensors. Defaults to "cpu".
        """
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['texts', 'task_labels']  # Attributes stored in the buffer

        # Initialize two buffers
        self.active_buffer = {'texts': [None] * buffer_size, 'task_labels': [None] * buffer_size}
        self.secondary_buffer = {'texts': [None] * buffer_size, 'task_labels': [None] * buffer_size}
        self.current_buffer = self.active_buffer  # Start with active buffer

        # Lock and thread for asynchronous saving
        self.save_lock = Lock()
        self.save_thread = None

    def to(self, device):
        """Move the buffer tensors to the specified device."""
        self.device = device
        return self

    def __len__(self):
        """Return the current number of elements in the active buffer."""
        return min(self.num_seen_examples, self.buffer_size)

    def add_data(self, texts: List[str], task_labels: List[str]):
        """
        Add data to the buffer using reservoir sampling and double buffering.

        Args:
            texts (List[str]): List of text strings.
            task_labels (List[str]): List of corresponding task labels.
        """
        for text, task_label in zip(texts, task_labels):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.current_buffer['texts'][index] = text
                self.current_buffer['task_labels'][index] = task_label

            # Check if the buffer is full and needs saving
            if self.num_seen_examples % self.buffer_size == 0:
                self._async_save_and_swap_buffers()

    def _async_save_and_swap_buffers(self):
        """Save the current buffer asynchronously and swap to the secondary buffer."""
        # Start asynchronous save of current buffer if no save is ongoing
        if self.save_thread is None or not self.save_thread.is_alive():
            buffer_to_save = self.current_buffer.copy()  # Copy buffer to avoid overwriting
            self.save_thread = Thread(target=self._save_buffer_to_disk, args=(buffer_to_save,))
            self.save_thread.start()

        # Swap buffers and clear the new active buffer for reuse
        self.current_buffer = self.secondary_buffer if self.current_buffer == self.active_buffer else self.active_buffer
        self._clear_buffer(self.current_buffer)

    def _clear_buffer(self, buffer):
        """Clear the specified buffer."""
        buffer['texts'] = [None] * self.buffer_size
        buffer['task_labels'] = [None] * self.buffer_size

    def _save_buffer_to_disk(self, buffer):
        """Save a buffer to disk."""
        directory = "buffer_save"
        os.makedirs(directory, exist_ok=True)

        # Save data attributes
        with self.save_lock:
            for attr_str in self.attributes:
                path = os.path.join(directory, f'{attr_str}.pt')
                torch.save(buffer[attr_str], path)

    def get_data(self, size: int, return_index=False) -> Tuple:
        """
        Sample a batch of data from the current buffer.

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
        sampled_texts = [self.current_buffer['texts'][i] for i in choice]
        sampled_task_labels = [self.current_buffer['task_labels'][i] for i in choice]

        if not return_index:
            return sampled_texts, sampled_task_labels 
        else:
            return torch.tensor(choice).to(self.device), sampled_texts, sampled_task_labels

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return self.num_seen_examples == 0

    def empty(self) -> None:
        """Empty both buffers."""
        self.num_seen_examples = 0
        self._clear_buffer(self.active_buffer)
        self._clear_buffer(self.secondary_buffer)

    def save(self, directory: str):
        """Save the buffer metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        torch.save({'buffer_size': self.buffer_size}, os.path.join(directory, 'metadata.pt'))

    def load(self, directory: str):
        """Load buffer metadata from disk."""
        metadata = torch.load(os.path.join(directory, 'metadata.pt'))
        self.buffer_size = metadata['buffer_size']
        self.num_seen_examples = metadata.get('num_seen_examples', 0)

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """Reservoir sampling strategy for adding data to the buffer."""
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1
