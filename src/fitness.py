import numpy as np

def total_sum(data: np.ndarray) -> float:
    return np.sum(data)

def total_positive_ratio(data: np.ndarray, epsilon: float = 1e-6) -> float:
    return np.sum(data > epsilon) / data.size
