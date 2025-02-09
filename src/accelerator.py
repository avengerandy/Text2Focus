from src.pareto import Solution, IParetoFront, ParetoFront
from src.sliding_window import IWindowGenerator, Window
import numpy as np
import random
from typing import Generator

class CoordinateTransformer:
    def __init__(self, original_width: int, original_height: int, resized_width: int, resized_height: int):
        self.scale_x = float(original_width) / resized_width
        self.scale_y = float(original_height) / resized_height

    def convert_resized_to_original(self, x: int, y: int, to_int=True) -> tuple:
        original_x = x * self.scale_x
        original_y = y * self.scale_y

        if to_int:
            return int(original_x), int(original_y)
        return original_x, original_y

    def convert_original_to_resized(self, x: int, y: int, to_int=True) -> tuple:
        resized_x = x / self.scale_x
        resized_y = y / self.scale_y

        if to_int:
            return int(resized_x), int(resized_y)
        return resized_x, resized_y

    def convert_original_ratio_to_resized(self, ratio_x: int, ratio_y: int, except_x: int, to_int=True) -> tuple:
        resized_x, resized_y = self.convert_original_to_resized(ratio_x, ratio_y, to_int=False)
        resized_y = resized_y * (except_x / resized_x)

        if to_int:
            return int(except_x), int(resized_y)
        return except_x, resized_y

    def convert_resized_ratio_to_original(self, ratio_x: int, ratio_y: int, except_x: int, to_int=True) -> tuple:
        original_x, original_y = self.convert_resized_to_original(ratio_x, ratio_y, to_int=False)
        original_y = original_y * (except_x / original_x)

        if to_int:
            return int(except_x), int(original_y)
        return except_x, original_y


class DividedParetoFront(IParetoFront):
    def __init__(self, solution_dimensions: int, num_subsets: int = 4):
        self.solution_dimensions = solution_dimensions
        self.num_subsets = num_subsets
        self.subsets = []
        self.unified_pareto_front = ParetoFront(solution_dimensions)
        self.is_updated = False

        for _ in range(num_subsets):
            self.subsets.append(ParetoFront(solution_dimensions))

    def add_solution(self, solution: Solution):
        subset_index = self._hash_solution(solution) % self.num_subsets
        self.subsets[subset_index].add_solution(solution)

        self.is_updated = True

    def get_pareto_solutions(self) -> list:
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_pareto_solutions()

    def get_elbow_point(self) -> Solution:
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_elbow_point()

    def get_point_by_weight(self, weight: list[float]) -> Solution:
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.get_point_by_weight(weight)

    def select_representative_solutions(self, num_solutions: int, random_state=42) -> list[Solution]:
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front.select_representative_solutions(num_solutions)

    def _get_pareto(self) -> np.ndarray:
        if self.is_updated:
            self._update_unified_pareto_front()

        return self.unified_pareto_front._get_pareto()

    def _update_unified_pareto_front(self):
        all_pareto_solutions = []
        for subset in self.subsets:
            all_pareto_solutions.extend(subset.get_pareto_solutions())

        self.unified_pareto_front = ParetoFront(self.solution_dimensions)
        for solution in all_pareto_solutions:
            self.unified_pareto_front.add_solution(solution)

        self.is_updated = False

    def _hash_solution(self, solution: Solution):
        return hash(tuple(solution.get_values()))


class GeneWindowGenerator(IWindowGenerator):
    """
    A window generator that uses genetic algorithm-like operations.
    """

    def __init__(
        self,
        image_matrix: np.ndarray,
        width_ratio: int,
        height_ratio: int,
        crossover_mutation_ratio: float = 0.8,
        min_population_size: int = 10,
        fitness_function_evaluations: int = 1000,
        random_seed: int = None  # Added random_seed for reproducibility
    ):
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
        Generate a random rectangle position and size.
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
        Generate a random window using a random rectangle.
        """
        i, j, width, height = self.generate_random_rectangle()
        sub_image_matrix = self.image_matrix[j : j + height, i : i + width]
        return Window(sub_image_matrix=sub_image_matrix, i=i, j=j, window_width=width, window_height=height)

    def crossover(self, parent1: Window, parent2: Window) -> Window:
        """
        Perform crossover between two parent windows to create a new child window.
        Introduces perturbation in the crossover process using weighted averages.
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
        window_width = int(weight1 * parent1.window_width + weight2 * parent2.window_width)

        weight1 = random.uniform(0.5, 0.9)
        weight2 = 1.0 - weight1
        window_height = int(weight1 * parent1.window_height + weight2 * parent2.window_height)

        # Ensure the new window is within bounds
        if i + window_width <= self.width and j + window_height <= self.height:
            return Window(
                sub_image_matrix=self.image_matrix[j : j + window_height, i : i + window_width],
                i=i,
                j=j,
                window_width=window_width,
                window_height=window_height,
            )
        else:
            return self.generate_random_window()

    def mutation(self, window: Window) -> Window:
        """
        Perform mutation on a window to create a new window.
        """
        i, j, window_width, window_height = self.generate_random_rectangle()
        i = int((window.i + i) / 2)
        j = int((window.j + j) / 2)
        window_width = int((window.window_width + window_width) / 2)
        window_height = int((window.window_height + window_height) / 2)

        if i + window_width < self.width and j + window_height < self.height:
            return Window(
                sub_image_matrix=self.image_matrix[j : j + window_height, i : i + window_width],
                i=i,
                j=j,
                window_width=window_width,
                window_height=window_height,
            )
        else:
            return self.generate_random_window()

    def generate_windows(self) -> Generator[Window, None, None]:
        """
        Generate a list of windows using genetic algorithm operations such as crossover and mutation.
        Ensures at least one operation occurs (crossover or mutation).
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
