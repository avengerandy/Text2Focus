import numpy as np
from abc import ABC, abstractmethod
from typing import Any

class Solution:
    def __init__(self, data: np.ndarray, metadata: Any = None):
        self._data = data
        self._metadata = metadata

    def get_values(self) -> np.ndarray:
        return self._data

    def is_dominated_by(self, other_solution: 'Solution') -> np.bool_:
        current_values = self.get_values()
        other_values = other_solution.get_values()

        all_less_equal = np.all(current_values <= other_values)
        any_strictly_less = np.any(current_values < other_values)

        return all_less_equal and any_strictly_less

    def get_metadata(self) -> Any:
        return self._metadata

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


class IParetoFront(ABC):
    @abstractmethod
    def add_solution(self, solution: Solution):
        pass

    @abstractmethod
    def get_pareto_solutions(self) -> list:
        pass

    @abstractmethod
    def get_elbow_point(self) -> Solution:
        pass

    @abstractmethod
    def get_point_by_weight(self, weight: list[float]) -> Solution:
        pass

    @abstractmethod
    def _get_pareto(self) -> np.ndarray:
        pass


class ParetoFront(IParetoFront):
    INITIAL_COLUMN_CAPACITY = 1000

    def __init__(self, solution_dimensions: int):
        self.matrix = DynamicRowMatrix(solution_dimensions, ParetoFront.INITIAL_COLUMN_CAPACITY)
        self.pareto_front_mask = DynamicRowMatrix(1, ParetoFront.INITIAL_COLUMN_CAPACITY)
        self._metadatas = []

    def add_solution(self, solution: Solution):
        is_dominated = False

        all_solutions_size = self.matrix.get_size()
        for i in range(all_solutions_size):
            if self.pareto_front_mask.get_data()[i][0] == 1:
                existing_solution = Solution(self.matrix.get_data()[i])
                if solution.is_dominated_by(existing_solution):
                    is_dominated = True
                    break

        if not is_dominated:
            self.matrix.add_row(solution.get_values())
            self.pareto_front_mask.add_row([1])
            self._metadatas.append(solution.get_metadata())

            for i in range(all_solutions_size):
                if self.pareto_front_mask.get_data()[i][0] == 1:
                    existing_solution = Solution(self.matrix.get_data()[i])
                    if existing_solution.is_dominated_by(solution):
                        self.pareto_front_mask.get_data()[i][0] = 0

    def get_pareto_solutions(self) -> list[Solution]:
        indexs = self._get_pareto_index()
        solutions = self.matrix.get_data()
        metadatas = self._metadatas
        return [Solution(solutions[row], metadatas[row]) for row in indexs]

    def get_elbow_point(self) -> Solution:
        if len(self._get_pareto()) == 1:
            return Solution(self._get_pareto()[0])

        pareto_values = self._normalize_pareto()
        diffs = np.diff(pareto_values, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        diffs_of_diffs = np.diff(distances)

        max_change_index = np.argmax(diffs_of_diffs) + 1
        best_elbow_point = self._get_pareto()[max_change_index]

        metadata = self._metadatas[self._get_pareto_index()[max_change_index]]

        return Solution(best_elbow_point, metadata)

    def get_point_by_weight(self, weight: list[float]) -> Solution:
        if len(self._get_pareto()) == 1:
            return Solution(self._get_pareto()[0])

        pareto_values = self._normalize_pareto()
        pareto_values = pareto_values * weight
        pareto_score = np.sum(pareto_values, axis=1)
        max_index = np.argmax(pareto_score)

        metadata = self._metadatas[self._get_pareto_index()[max_index]]

        return Solution(self._get_pareto()[max_index], metadata)


    def _normalize_pareto(self) -> np.ndarray:
        pareto_front = self._get_pareto()
        min_vals = np.min(pareto_front, axis=0)
        max_vals = np.max(pareto_front, axis=0)
        return (pareto_front - min_vals) / (max_vals - min_vals + 1e-8)

    def _get_pareto(self) -> np.ndarray:
        pareto_front = self.matrix.get_data()
        pareto_mask = self.pareto_front_mask.get_data()

        pareto_mask = np.squeeze(pareto_mask)

        return pareto_front[pareto_mask == 1]

    def _get_pareto_index(self) -> np.ndarray:
        pareto_mask = self.pareto_front_mask.get_data()
        pareto_mask = np.squeeze(pareto_mask)

        return np.where(pareto_mask == 1)[0]
