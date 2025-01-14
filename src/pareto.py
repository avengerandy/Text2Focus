import numpy as np

class Solution:
    def __init__(self, data: np.ndarray):
        self._data = data

    def get_values(self) -> np.ndarray:
        return self._data

    def is_dominated_by(self, other_solution: 'Solution') -> np.bool_:
        current_values = self.get_values()
        other_values = other_solution.get_values()

        all_less_equal = np.all(current_values <= other_values)
        any_strictly_less = np.any(current_values < other_values)

        return all_less_equal and any_strictly_less

    def __repr__(self) -> str:
        if len(self._data) < 5:
            return f"Solution({self._data})"
        return f"Solution({self._data[:5]}...)"

class DynamicRowMatrix:
    EXPANSION_FACTOR = 2

    def __init__(self, initial_column_capacity, initial_row_capacity: int = 10):
        self._row_capacity = initial_row_capacity
        self._column_capacity = initial_column_capacity
        self._matrix = np.zeros((self._row_capacity, self._column_capacity), dtype=np.float64)
        self._size = 0

    def add_row(self, new_data: np.ndarray):
        if self._size == self._row_capacity:
            self._increase_row_capacity()

        self._matrix[self._size] = new_data
        self._size += 1

    def get_size(self):
        return self._size

    def get_data(self):
        return self._matrix[:self._size]

    def _increase_row_capacity(self):
        self._row_capacity = self._row_capacity * DynamicRowMatrix.EXPANSION_FACTOR

        new_matrix = np.zeros((self._row_capacity, self._column_capacity), dtype=np.float64)
        new_matrix[:self._size] = self._matrix

        self._matrix = new_matrix
