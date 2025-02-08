"""
This module provides classes and methods for managing Pareto front solutions.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


class Solution:
    """
    Represents a solution with its associated values and optional metadata.

    Attributes:
        data (np.ndarray): The values of the solution (objectives, decision
        variables, etc.).
        metadata (Any): Optional additional metadata associated with the solution.
    """

    def __init__(self, data: np.ndarray, metadata: Any = None):
        """
        Initializes a Solution instance with given values and optional metadata.

        Parameters:
            data (np.ndarray): The values of the solution.
            metadata (Any): Optional metadata associated with the solution.
        """
        self._data = data
        self._metadata = metadata

    def get_values(self) -> np.ndarray:
        """
        Returns the solution's values.

        Returns:
            np.ndarray: The values of the solution (e.g., objective values).
        """
        return self._data

    def is_dominated_by(self, other_solution: "Solution") -> np.bool_:
        """
        Checks if the current solution is dominated by another solution.

        A solution is dominated if all of its values are less than or equal to
        the corresponding values of another solution, and at least one value is
        strictly less.

        Parameters:
            other_solution (Solution): The other solution to compare against.

        Returns:
            np.bool_: True if the current solution is dominated by the other solution.
        """
        current_values = self.get_values()
        other_values = other_solution.get_values()

        all_less_equal = np.all(current_values <= other_values)
        any_strictly_less = np.any(current_values < other_values)

        return all_less_equal and any_strictly_less

    def get_metadata(self) -> Any:
        """
        Returns the solution's metadata.

        Returns:
            Any: The metadata associated with the solution.
        """
        return self._metadata

    def __repr__(self) -> str:
        """
        Provides a string representation of the solution, showing only the first
        5 elements if the data is long.

        Returns:
            str: String representation of the solution, showing first few values.
        """
        if len(self._data) < 5:
            return f"Solution({self._data})"
        return f"Solution({self._data[:5]}...)"


class DynamicRowMatrix:
    """
    A dynamic matrix that can resize its rows and holds numerical data.

    Attributes:
        row_capacity (int): Initial number of rows.
        column_capacity (int): Initial number of columns.
        matrix (np.ndarray): The underlying data matrix (2D numpy array).
        size (int): Current number of rows in the matrix.
    """

    EXPANSION_FACTOR = 2

    def __init__(self, initial_column_capacity, initial_row_capacity: int = 10):
        """
        Initializes a DynamicRowMatrix with specified initial row and column
        capacities.

        Parameters:
            initial_column_capacity (int): Initial number of columns.
            initial_row_capacity (int): Initial number of rows (default 10).
        """
        self._row_capacity = initial_row_capacity
        self._column_capacity = initial_column_capacity
        self._matrix = np.zeros(
            (self._row_capacity, self._column_capacity), dtype=np.float64
        )
        self._size = 0

    def add_row(self, new_data: np.ndarray):
        """
        Adds a new row of data to the matrix. Expands the matrix if the row
        capacity is exceeded.

        Parameters:
            new_data (np.ndarray): The data to be added as a new row.
        """
        if self._size == self._row_capacity:
            self._increase_row_capacity()

        self._matrix[self._size] = new_data
        self._size += 1

    def get_size(self):
        """
        Returns the current size (number of rows) of the matrix.

        Returns:
            int: The current number of rows in the matrix.
        """
        return self._size

    def get_data(self):
        """
        Returns the data in the matrix up to the current size.

        Returns:
            np.ndarray: The matrix data up to the current size.
        """
        return self._matrix[: self._size]

    def _increase_row_capacity(self):
        """
        Increases the row capacity of the matrix by a factor of EXPANSION_FACTOR
        and resizes the matrix.
        """
        self._row_capacity = self._row_capacity * DynamicRowMatrix.EXPANSION_FACTOR

        new_matrix = np.zeros(
            (self._row_capacity, self._column_capacity), dtype=np.float64
        )
        new_matrix[: self._size] = self._matrix

        self._matrix = new_matrix


class IParetoFront(ABC):
    """
    Interface for managing a Pareto front of solutions.

    This interface defines essential operations for managing Pareto optimal
    solutions in multi-objective optimization.
    """

    @abstractmethod
    def add_solution(self, solution: Solution):
        """
        Adds a solution to the Pareto front.

        Parameters:
            solution (Solution): The solution to be added to the Pareto front.
        """

    @abstractmethod
    def get_pareto_solutions(self) -> list:
        """
        Returns the Pareto optimal solutions.

        Returns:
            list: A list of Pareto optimal solutions.
        """

    @abstractmethod
    def get_elbow_point(self) -> Solution:
        """
        Returns the elbow point from the Pareto front, which is typically the
        solution that balances the trade-off between conflicting objectives.

        Returns:
            Solution: The solution corresponding to the elbow point.
        """

    @abstractmethod
    def get_point_by_weight(self, weight: list[float]) -> Solution:
        """
        Returns a solution from the Pareto front based on a weight vector.

        Parameters:
            weight (list[float]): A list of weights for each objective.

        Returns:
            Solution: The weighted solution from the Pareto front.
        """

    @abstractmethod
    def select_representative_solutions(
        self, num_solutions: int, random_state=42
    ) -> list[Solution]:
        """
        Selects representative solutions from the Pareto front.

        Parameters:
            num_solutions (int): Number of representative solutions to select.
            random_state (int): Random seed for clustering (default is 42).

        Returns:
            list: A list of selected representative solutions.
        """

    @abstractmethod
    def _get_pareto(self) -> np.ndarray:
        """
        Returns the Pareto front data.

        Returns:
            np.ndarray: Pareto front data as a numpy array.
        """


class ParetoFront(IParetoFront):
    """
    Represents a Pareto front, managing Pareto optimal solutions and providing
    methods for analyzing and selecting solutions.

    Attributes:
        matrix (DynamicRowMatrix): The matrix holding the Pareto front solutions.
        pareto_front_mask (DynamicRowMatrix): A mask to track which solutions
        are part of the Pareto front.
        _metadatas (list): Metadata associated with the Pareto front solutions.
    """

    INITIAL_COLUMN_CAPACITY = 1000

    def __init__(self, solution_dimensions: int):
        """
        Initializes a Pareto front for a given number of solution dimensions.

        Parameters:
            solution_dimensions (int): The number of dimensions of each solution.
        """
        self.matrix = DynamicRowMatrix(
            solution_dimensions, ParetoFront.INITIAL_COLUMN_CAPACITY
        )
        self.pareto_front_mask = DynamicRowMatrix(
            1, ParetoFront.INITIAL_COLUMN_CAPACITY
        )
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
            return Solution(
                self._get_pareto()[0], self._metadatas[self._get_pareto_index()[0]]
            )

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
            return Solution(
                self._get_pareto()[0], self._metadatas[self._get_pareto_index()[0]]
            )

        pareto_values = self._normalize_pareto()
        pareto_values = pareto_values * weight
        pareto_score = np.sum(pareto_values, axis=1)
        max_index = np.argmax(pareto_score)

        metadata = self._metadatas[self._get_pareto_index()[max_index]]

        return Solution(self._get_pareto()[max_index], metadata)

    def select_representative_solutions(
        self, num_solutions: int, random_state=42
    ) -> list[Solution]:
        """
        Selects representative solutions from the Pareto front using clustering
        methods.

        Parameters:
            num_solutions (int): The number of representative solutions to select.
            random_state (int): Random seed for clustering (default is 42).

        Returns:
            list: A list of representative solutions selected from the Pareto front.
        """
        pareto_front = self._normalize_pareto()

        lower_bound = 0.1
        upper_bound = 0.9
        valid_indices = []

        for i, solution in enumerate(pareto_front):
            if np.all(solution >= lower_bound) and np.all(solution <= upper_bound):
                valid_indices.append(i)

        if len(valid_indices) < num_solutions:
            valid_indices = range(len(pareto_front))

        if len(pareto_front[valid_indices]) <= num_solutions:
            solutions = self.get_pareto_solutions()
            return [solutions[i] for i in valid_indices]

        kmeans = KMeans(n_clusters=num_solutions, random_state=random_state)
        kmeans.fit(pareto_front[valid_indices])

        selected_solutions = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(pareto_front[valid_indices] - center, axis=1)
            closest_index = np.argmin(distances)
            selected_solutions.append(
                self.get_pareto_solutions()[valid_indices[closest_index]]
            )

        return selected_solutions

    def _normalize_pareto(self) -> np.ndarray:
        pareto_front = self._get_pareto()
        min_vals = np.min(pareto_front, axis=0)
        max_vals = np.max(pareto_front, axis=0)
        return (pareto_front - min_vals) / (max_vals - min_vals + 1e-8)

    def _get_pareto(self) -> np.ndarray:
        pareto_front = self.matrix.get_data()
        pareto_mask = self.pareto_front_mask.get_data()

        if pareto_mask.shape[0] == 1:
            return pareto_front

        pareto_mask = np.squeeze(pareto_mask)
        return pareto_front[pareto_mask == 1]

    def _get_pareto_index(self) -> np.ndarray:
        pareto_mask = self.pareto_front_mask.get_data()

        if pareto_mask.shape[0] == 1:
            return np.array([0])

        pareto_mask = np.squeeze(pareto_mask)
        return np.where(pareto_mask == 1)[0]
