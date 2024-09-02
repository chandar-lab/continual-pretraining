import numpy as np
from typing import List, Dict, Tuple

def calculate_metrics(accuracy_matrix: List[List[float]], num_tasks_seen: int) -> Tuple[float, float]:
    """
    Calculate plasticity and forgetting metrics from the accuracy matrix.
    
    Args:
    - accuracy_matrix: A matrix where each row represents accuracies across all tasks
                       after training on a specific task.
    - num_tasks_seen: Number of tasks seen so far (including the current task)
    
    Returns:
    - Tuple of (plasticity, forgetting)
    """
    plasticity_values = []
    forgetting_values = []
    
    # Plasticity: improvement on new task
    plasticity = accuracy_matrix[-1][num_tasks_seen-1] - accuracy_matrix[-2][num_tasks_seen-1]
    plasticity_values.append(plasticity)
    
    # Forgetting: performance decrease on previous tasks
    for j in range(num_tasks_seen - 1):
        forgetting = max(0, accuracy_matrix[-2][j] - accuracy_matrix[-1][j])
        forgetting_values.append(forgetting)
    
    avg_plasticity = sum(plasticity_values) / len(plasticity_values)
    avg_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0
    
    return avg_plasticity, avg_forgetting