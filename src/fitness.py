import numpy as np

def total_sum(data: np.ndarray) -> float:
    return np.sum(data)

def total_positive_ratio(data: np.ndarray, epsilon: float = 1e-6) -> float:
    return np.sum(data > epsilon) / data.size

def total_cut_ratio(data: np.ndarray) -> float:
    top_row = data[0, :]
    bottom_row = data[-1, :]
    left_column = data[1:-1, 0]
    right_column = data[1:-1, -1]
    boundary_sum = np.sum(top_row) + np.sum(bottom_row) + np.sum(left_column) + np.sum(right_column)

    return -(boundary_sum / (data.shape[0] + data.shape[1] + data.shape[0] + data.shape[1] - 4))
