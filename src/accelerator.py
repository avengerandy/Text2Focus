"""
This module contains a set of classes to accelerate the optimization process.
"""

import random
from typing import Generator

import numpy as np

from src.pareto import IParetoFront, ParetoFront, Solution
from src.sliding_window import IWindowGenerator, Window


class CoordinateTransformer:
    """
    A class to transform coordinates between original and resized dimensions.
    For example, converting coordinates from a resized image back to the original image.

    Attributes:
        scale_x (float): The scaling factor for the x-coordinate.
        scale_y (float): The scaling factor for the y-coordinate.
    """

    def __init__(
        self,
        original_width: int,
        original_height: int,
        resized_width: int,
        resized_height: int,
    ):
        """
        Initializes the coordinate transformer by calculating the scaling factors.
        The scaling factors determine how much the original and resized dimensions differ.

        Parameters:
            original_width (int): The width of the original image.
            original_height (int): The height of the original image.
            resized_width (int): The width of the resized image.
            resized_height (int): The height of the resized image.
        """
        self.scale_x = float(original_width) / resized_width
        self.scale_y = float(original_height) / resized_height

    def convert_resized_to_original(self, x: int, y: int, to_int=True) -> tuple:
        """
        Converts coordinates from the resized image back to the original image's coordinates.

        Parameters:
            x (int): The x-coordinate in the resized image.
            y (int): The y-coordinate in the resized image.
            to_int (bool): Whether to return the result as integers (default is True).

        Returns:
            tuple: The (x, y) coordinates in the original image.
        """
        original_x = x * self.scale_x
        original_y = y * self.scale_y

        if to_int:
            return int(original_x), int(original_y)
        return original_x, original_y

    def convert_original_to_resized(self, x: int, y: int, to_int=True) -> tuple:
        """
        Converts coordinates from the original image to the resized image's coordinates.

        Parameters:
            x (int): The x-coordinate in the original image.
            y (int): The y-coordinate in the original image.
            to_int (bool): Whether to return the result as integers (default is True).

        Returns:
            tuple: The (x, y) coordinates in the resized image.
        """
        resized_x = x / self.scale_x
        resized_y = y / self.scale_y

        if to_int:
            return int(resized_x), int(resized_y)
        return resized_x, resized_y

    def convert_original_ratio_to_resized(
        self, ratio_x: int, ratio_y: int, except_x: int, to_int=True
    ) -> tuple:
        """
        Converts the ratio of the original image to the resized image based on a specified
        x-coordinate.

        Parameters:
            ratio_x (int): The x-coordinate ratio in the original image.
            ratio_y (int): The y-coordinate ratio in the original image.
            except_x (int): The fixed x-coordinate in the resized image.
            to_int (bool): Whether to return the result as integers (default is True).

        Returns:
            tuple: The (except_x, resized_y) coordinates in the resized image.
        """
        resized_x, resized_y = self.convert_original_to_resized(
            ratio_x, ratio_y, to_int=False
        )
        resized_y = resized_y * (except_x / resized_x)

        if to_int:
            return int(except_x), int(resized_y)
        return except_x, resized_y

    def convert_resized_ratio_to_original(
        self, ratio_x: int, ratio_y: int, except_x: int, to_int=True
    ) -> tuple:
        """
        Converts the ratio of the resized image to the original image based on a specified
        x-coordinate.

        Parameters:
            ratio_x (int): The x-coordinate ratio in the resized image.
            ratio_y (int): The y-coordinate ratio in the resized image.
            except_x (int): The fixed x-coordinate in the original image.
            to_int (bool): Whether to return the result as integers (default is True).

        Returns:
            tuple: The (except_x, original_y) coordinates in the original image.
        """
        original_x, original_y = self.convert_resized_to_original(
            ratio_x, ratio_y, to_int=False
        )
        original_y = original_y * (except_x / original_x)

        if to_int:
            return int(except_x), int(original_y)
        return except_x, original_y


class DividedParetoFront(IParetoFront):
    """
    A class that manages a divided Pareto front by splitting the solutions into subsets and
    combining them later.

    Attributes:
        solution_dimensions (int): The number of dimensions of the solution space.
        num_subsets (int): The number of subsets to divide the Pareto front into.
        subsets (list): A list of ParetoFront objects representing each subset.
        unified_pareto_front (ParetoFront): The unified Pareto front after combining all subsets.
        is_updated (bool): A flag indicating whether the unified Pareto front needs updating.
    """

    def __init__(self, solution_dimensions: int, num_subsets: int = 4):
        """
        Initializes the DividedParetoFront with the number of subsets and solution dimensions.

        Parameters:
            solution_dimensions (int): The number of dimensions of the solution space.
            num_subsets (int): The number of subsets to divide the Pareto front into (default is 4).
        """
        self.solution_dimensions = solution_dimensions
        self.num_subsets = num_subsets
        self.subsets = []
        self.unified_pareto_front = ParetoFront(solution_dimensions)
        self.is_updated = False

        for _ in range(num_subsets):
            self.subsets.append(ParetoFront(solution_dimensions))

    def add_solution(self, solution: Solution):
        """
        Adds a solution to the appropriate subset based on a hash function.

        Parameters:
            solution (Solution): The solution to add to the Pareto front.
        """
        subset_index = self._hash_solution(solution) % self.num_subsets
        self.subsets[subset_index].add_solution(solution)

        self.is_updated = True

    def get_pareto_solutions(self) -> list:
        """
        Returns the Pareto solutions after updating the unified Pareto front if necessary.

        Returns:
            list: A list of Pareto optimal solutions.
        """
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_pareto_solutions()

    def get_elbow_point(self) -> Solution:
        """
        Returns the elbow point of the Pareto front after updating the unified Pareto front
        if necessary.

        Returns:
            Solution: The elbow point solution.
        """
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_elbow_point()

    def get_point_by_weight(self, weight: list[float]) -> Solution:
        """
        Returns a point from the Pareto front based on a specified weight distribution after
        updating the unified Pareto front if necessary.

        Parameters:
            weight (list): The weight distribution used to select a specific point.

        Returns:
            Solution: The solution at the weighted point.
        """
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_point_by_weight(weight)

    def select_representative_solutions(
        self, num_solutions: int, random_state=42
    ) -> list[Solution]:
        """
        Selects representative solutions from the Pareto front after updating the unified
        Pareto front if necessary.

        Parameters:
            num_solutions (int): The number of solutions to select.
            random_state (int): The random seed for reproducibility (default is 42).

        Returns:
            list: A list of representative solutions.
        """
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.select_representative_solutions(num_solutions)

    def get_pareto(self) -> np.ndarray:
        """
        Returns the raw Pareto front as a numpy array after updating the unified
        Pareto front if necessary.

        Returns:
            np.ndarray: The raw Pareto front data.
        """
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_pareto()

    def get_percentile_rank(self, solution: Solution) -> list[float]:
        """
        Calculates the Percentile Rank (PR) of the given solution for each dimension
        in the unified Pareto front.

        Parameters:
            solution (Solution): The solution for which to calculate PR values.

        Returns:
            list[float]: A list of PR values for each dimension.
        """
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_percentile_rank(solution)

    def _update_unified_pareto_front(self):
        """
        Updates the unified Pareto front by combining all the subsets.
        """
        all_pareto_solutions = []
        for subset in self.subsets:
            all_pareto_solutions.extend(subset.get_pareto_solutions())

        self.unified_pareto_front = ParetoFront(self.solution_dimensions)
        for solution in all_pareto_solutions:
            self.unified_pareto_front.add_solution(solution)

        self.is_updated = False

    def _hash_solution(self, solution: Solution):
        """
        Hashes the solution into a unique index for the subset division.

        Parameters:
            solution (Solution): The solution to hash.

        Returns:
            int: The hash value of the solution.
        """
        return hash(tuple(solution.get_values()))


class GeneWindowGenerator(IWindowGenerator):
    """
    A window generator that uses genetic algorithm-like operations to generate image windows.

    Attributes:
        image_matrix (np.ndarray): The image data from which to generate windows.
        width_ratio (int): The ratio of width used for window generation.
        height_ratio (int): The ratio of height used for window generation.
        width (int): The width of the image.
        height (int): The height of the image.
        crossover_mutation_ratio (float): The probability of performing crossover or mutation.
        min_population_size (int): Minimum number of windows to keep in the population.
    """

    def __init__(
        self,
        image_matrix: np.ndarray,
        width_ratio: int,
        height_ratio: int,
        crossover_mutation_ratio: float = 0.8,
        min_population_size: int = 10,
        fitness_function_evaluations: int = 1000,
        random_seed: int = None,  # Added random_seed for reproducibility
    ):
        """
        Initializes the GeneWindowGenerator with genetic algorithm parameters and image dimensions.

        Parameters:
            image_matrix (np.ndarray): The image data from which to generate windows.
            width_ratio (int): The ratio of width used for window generation.
            height_ratio (int): The ratio of height used for window generation.
            crossover_mutation_ratio (float):
                The probability of performing crossover or mutation (default 0.8).
            min_population_size (int):
                Minimum number of windows to keep in the population (default 10).
            fitness_function_evaluations (int):
                The number of fitness evaluations for the genetic algorithm (default 1000).
            random_seed (int): The random seed for reproducibility (optional).
        """
        self.image_matrix = image_matrix
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
        self.width, self.height = image_matrix.shape
        self.crossover_mutation_ratio = crossover_mutation_ratio
        self.min_population_size = min_population_size
        self.fitness_function_evaluations = fitness_function_evaluations
        self.population = []

        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

    def generate_random_rectangle(self) -> tuple:
        """
        Generates a random rectangle within the bounds of the image.

        Returns:
            tuple: The coordinates and size of the rectangle (x, y, width, height).
        """
        rect_width = random.uniform(1, self.width)
        rect_height = max((rect_width * self.height_ratio) / self.width_ratio, 1.0)

        if rect_height > self.height:
            rect_height = self.height
            rect_width = max((rect_height * self.width_ratio) / self.height_ratio, 1.0)

        x = random.uniform(0, self.width - rect_width)
        y = random.uniform(0, self.height - rect_height)

        return int(x), int(y), int(rect_width), int(rect_height)

    def generate_random_window(self) -> Window:
        """
        Generates a random window based on a random rectangle.

        Returns:
            Window: The generated window object.
        """
        i, j, width, height = self.generate_random_rectangle()
        sub_image_matrix = self.image_matrix[j : j + height, i : i + width]
        return Window(
            sub_image_matrix=sub_image_matrix,
            i=i,
            j=j,
            window_width=width,
            window_height=height,
        )

    def crossover(self, parent1: Window, parent2: Window) -> Window:
        """
        Performs crossover between two parent windows to create a child window.

        Parameters:
            parent1 (Window): The first parent window.
            parent2 (Window): The second parent window.

        Returns:
            Window: The child window created by crossover.
        """
        # Weighted average of the parent window parameters with random weights
        weight1 = random.uniform(0.5, 0.9)
        weight2 = 1.0 - weight1
        i = int(weight1 * parent1.i + weight2 * parent2.i)

        weight1 = random.uniform(0.5, 0.9)
        weight2 = 1.0 - weight1
        j = int(weight1 * parent1.j + weight2 * parent2.j)

        weight1 = random.uniform(0.5, 0.9)
        weight2 = 1.0 - weight1
        window_width = int(
            weight1 * parent1.window_width + weight2 * parent2.window_width
        )

        weight1 = random.uniform(0.5, 0.9)
        weight2 = 1.0 - weight1
        window_height = int(
            weight1 * parent1.window_height + weight2 * parent2.window_height
        )

        # Ensure the new window is within bounds
        if i + window_width <= self.width and j + window_height <= self.height:
            return Window(
                sub_image_matrix=self.image_matrix[
                    j : j + window_height, i : i + window_width
                ],
                i=i,
                j=j,
                window_width=window_width,
                window_height=window_height,
            )

        return self.generate_random_window()

    def mutation(self, window: Window) -> Window:
        """
        Performs mutation on a window to create a new window.

        Parameters:
            window (Window): The window to mutate.

        Returns:
            Window: The mutated window.
        """
        i, j, window_width, window_height = self.generate_random_rectangle()
        i = int((window.i + i) / 2)
        j = int((window.j + j) / 2)
        window_width = int((window.window_width + window_width) / 2)
        window_height = int((window.window_height + window_height) / 2)

        if i + window_width < self.width and j + window_height < self.height:
            return Window(
                sub_image_matrix=self.image_matrix[
                    j : j + window_height, i : i + window_width
                ],
                i=i,
                j=j,
                window_width=window_width,
                window_height=window_height,
            )

        return self.generate_random_window()

    def generate_windows(self) -> Generator[Window, None, None]:
        """
        Generates a list of windows using genetic algorithm operations such as crossover and
        mutation. Ensures at least one operation occurs (crossover or mutation).

        Returns:
            Generator[Window, None, None]: A generator of windows.
        """
        for _ in range(self.fitness_function_evaluations):
            if len(self.population) < self.min_population_size:
                yield self.generate_random_window()
                continue

            parent1, parent2 = random.sample(self.population, 2)

            # Ensure that at least one operation happens (either crossover or mutation)
            if random.random() < self.crossover_mutation_ratio:
                yield self.crossover(parent1, parent2)
            else:
                yield self.mutation(parent1)
