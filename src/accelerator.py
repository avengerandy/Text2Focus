from src.pareto import Solution, IParetoFront, ParetoFront
import numpy as np


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
