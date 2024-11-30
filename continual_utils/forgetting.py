import pickle

# Function to save forgetting metrics to disk
def save_forgetting_metrics(forgetting_metrics, task_id, current_performance, best_performances, filepath="forgetting/forgetting_metrics.pkl"):
    if task_id not in forgetting_metrics:
        forgetting_metrics[task_id] = {
            "best_performance": float('inf'),
            "forgetting_values": []
        }

    # Get the best performance for the current task
    best_performance = forgetting_metrics[task_id]["best_performance"]
    if current_performance < best_performance:
        forgetting_metrics[task_id]["best_performance"] = current_performance
        forgetting_value = 0  # No forgetting if current performance is best
    else:
        forgetting_value = best_performance - current_performance

    # Append forgetting value for the current task
    forgetting_metrics[task_id]["forgetting_values"].append(forgetting_value)

    # Calculate average forgetting for this task
    average_forgetting = sum(forgetting_metrics[task_id]["forgetting_values"]) / len(forgetting_metrics[task_id]["forgetting_values"])

    # Save forgetting metrics to disk
    with open(filepath, 'wb') as f:
        pickle.dump(forgetting_metrics, f)

    return forgetting_metrics, forgetting_value, average_forgetting


# Function to load forgetting metrics from disk
def load_forgetting_metrics(filepath="forgetting_metrics.pkl"):
    try:
        with open(filepath, 'rb') as f:
            forgetting_metrics = pickle.load(f)
        return forgetting_metrics
    except FileNotFoundError:
        print(f"No previous forgetting metrics found at {filepath}. Starting fresh.")
        return {}