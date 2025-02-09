import unittest

import numpy as np

from src.accelerator import (
    CoordinateTransformer,
    DividedParetoFront,
    GeneWindowGenerator,
)
from src.pareto import IParetoFront, Solution
from src.sliding_window import IWindowGenerator, Window


class TestCoordinateTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = CoordinateTransformer(
            original_width=1000,
            original_height=500,
            resized_width=256,
            resized_height=256,
        )

    def test_convert_resized_to_original(self):
        resized_x, resized_y = 128, 128
        original_x, original_y = self.transformer.convert_resized_to_original(
            resized_x, resized_y
        )
        self.assertEqual(original_x, 500)
        self.assertEqual(original_y, 250)

        original_x, original_y = self.transformer.convert_resized_to_original(
            resized_x, resized_y, to_int=False
        )
        self.assertEqual(original_x, 500.0)
        self.assertEqual(original_y, 250.0)

    def test_convert_original_to_resized(self):
        original_x, original_y = 500, 250
        resized_x, resized_y = self.transformer.convert_original_to_resized(
            original_x, original_y
        )
        self.assertEqual(resized_x, 128)
        self.assertEqual(resized_y, 128)

        resized_x, resized_y = self.transformer.convert_original_to_resized(
            original_x, original_y, to_int=False
        )
        self.assertEqual(resized_x, 128.0)
        self.assertEqual(resized_y, 128.0)

    def test_convert_original_ratio_to_resized(self):
        ratio_x, ratio_y, except_x = 16, 9, 256
        resized_x, resized_y = self.transformer.convert_original_ratio_to_resized(
            ratio_x, ratio_y, except_x
        )
        self.assertEqual(resized_x, 256)
        self.assertEqual(resized_y, 288)

        resized_x, resized_y = self.transformer.convert_original_ratio_to_resized(
            ratio_x, ratio_y, except_x, to_int=False
        )
        self.assertEqual(resized_x, 256.0)
        self.assertEqual(resized_y, 288.0)

    def test_convert_resized_ratio_to_original(self):
        ratio_x, ratio_y, except_x = 16, 9, 256
        original_x, original_y = self.transformer.convert_resized_ratio_to_original(
            ratio_x, ratio_y, except_x
        )
        self.assertEqual(original_x, 256)
        self.assertEqual(original_y, 72)

        original_x, original_y = self.transformer.convert_resized_ratio_to_original(
            ratio_x, ratio_y, except_x, to_int=False
        )
        self.assertEqual(original_x, 256.0)
        self.assertEqual(original_y, 72.0)


class TestDividedParetoFront(unittest.TestCase):

    def setUp(self):
        self.pareto_front = DividedParetoFront(solution_dimensions=2)

    def test_is_pareto_front_interface(self):
        self.assertTrue(issubclass(DividedParetoFront, IParetoFront))

    def test_add_solution(self):
        solution_1 = Solution(np.array([1, 2], dtype=np.float64), metadata="metadata1")
        solution_2 = Solution(np.array([2, 1], dtype=np.float64), metadata="metadata2")

        self.pareto_front.add_solution(solution_1)
        self.pareto_front.add_solution(solution_2)

        pareto_solutions = self.pareto_front.get_pareto_solutions()
        self.assertEqual(len(pareto_solutions), 2)

    def test_get_pareto_solutions(self):
        solution_1 = Solution(np.array([1, 2], dtype=np.float64), metadata="metadata1")
        solution_2 = Solution(np.array([2, 1], dtype=np.float64), metadata="metadata2")
        solution_3 = Solution(np.array([3, 1], dtype=np.float64), metadata="metadata3")

        self.pareto_front.add_solution(solution_1)
        self.pareto_front.add_solution(solution_2)
        self.pareto_front.add_solution(solution_3)

        pareto_solutions = self.pareto_front.get_pareto_solutions()

        self.assertEqual(len(pareto_solutions), 2)
        self.assertTrue(
            any(
                np.array_equal(solution_1.get_values(), p.get_values())
                for p in pareto_solutions
            )
        )
        self.assertIn(
            solution_1.get_metadata(), [p.get_metadata() for p in pareto_solutions]
        )
        self.assertTrue(
            any(
                np.array_equal(solution_3.get_values(), p.get_values())
                for p in pareto_solutions
            )
        )
        self.assertIn(
            solution_3.get_metadata(), [p.get_metadata() for p in pareto_solutions]
        )
        self.assertFalse(
            any(
                np.array_equal(solution_2.get_values(), p.get_values())
                for p in pareto_solutions
            )
        )
        self.assertNotIn(
            solution_2.get_metadata(), [p.get_metadata() for p in pareto_solutions]
        )

    def test_get_elbow_point(self):
        solution_0 = Solution(np.array([9, 8], dtype=np.float64), metadata="metadata0")
        solution_1 = Solution(np.array([10, 9], dtype=np.float64), metadata="metadata1")
        solution_2 = Solution(np.array([20, 8], dtype=np.float64), metadata="metadata2")
        solution_3 = Solution(np.array([40, 1], dtype=np.float64), metadata="metadata3")
        solution_4 = Solution(np.array([50, 2], dtype=np.float64), metadata="metadata4")

        # one solution
        self.pareto_front.add_solution(solution_0)
        elbow_point = self.pareto_front.get_elbow_point()
        self.assertTrue(
            np.array_equal(solution_0.get_values(), elbow_point.get_values())
        )
        self.assertEqual(elbow_point.get_metadata(), "metadata0")

        # one pareto solution
        self.pareto_front.add_solution(solution_1)
        elbow_point = self.pareto_front.get_elbow_point()
        self.assertTrue(
            np.array_equal(solution_1.get_values(), elbow_point.get_values())
        )
        self.assertEqual(elbow_point.get_metadata(), "metadata1")

        # multiple pareto solutions
        self.pareto_front.add_solution(solution_2)
        self.pareto_front.add_solution(solution_3)
        self.pareto_front.add_solution(solution_4)

        elbow_point = self.pareto_front.get_elbow_point()
        self.assertTrue(
            np.array_equal(solution_2.get_values(), elbow_point.get_values())
        )
        self.assertEqual(elbow_point.get_metadata(), "metadata2")

    def test_get_point_by_weight(self):
        solution_0 = Solution(np.array([0, 0], dtype=np.float64), metadata="metadata0")
        solution_1 = Solution(np.array([1, 3], dtype=np.float64), metadata="metadata1")
        solution_2 = Solution(np.array([2, 2], dtype=np.float64), metadata="metadata2")
        solution_3 = Solution(np.array([5, 1], dtype=np.float64), metadata="metadata3")

        # one solution
        self.pareto_front.add_solution(solution_0)
        weight = [0.5, 0.5]
        point = self.pareto_front.get_point_by_weight(weight)
        self.assertTrue(np.array_equal(solution_0.get_values(), point.get_values()))
        self.assertEqual(point.get_metadata(), "metadata0")

        # one pareto solution
        self.pareto_front.add_solution(solution_1)
        weight = [0.5, 0.5]
        point = self.pareto_front.get_point_by_weight(weight)
        self.assertTrue(np.array_equal(solution_1.get_values(), point.get_values()))
        self.assertEqual(point.get_metadata(), "metadata1")

        # multiple pareto solutions
        self.pareto_front.add_solution(solution_2)
        self.pareto_front.add_solution(solution_3)

        weight = [0.5, 0.5]
        point = self.pareto_front.get_point_by_weight(weight)
        self.assertTrue(np.array_equal(solution_3.get_values(), point.get_values()))
        self.assertEqual(point.get_metadata(), "metadata3")

        weight = [0.1, 0.9]
        point = self.pareto_front.get_point_by_weight(weight)
        self.assertTrue(np.array_equal(solution_1.get_values(), point.get_values()))
        self.assertEqual(point.get_metadata(), "metadata1")

    def test_select_representative_solutions(self):

        solution_1 = Solution(
            np.array([1.1, 3.9], dtype=np.float64), metadata="metadata1"
        )
        solution_2 = Solution(
            np.array([1.0, 4.0], dtype=np.float64), metadata="metadata2"
        )
        solution_3 = Solution(
            np.array([0.9, 4.1], dtype=np.float64), metadata="metadata3"
        )
        solution_4 = Solution(
            np.array([3.9, 1.1], dtype=np.float64), metadata="metadata4"
        )
        solution_5 = Solution(
            np.array([4.0, 1.0], dtype=np.float64), metadata="metadata5"
        )
        solution_6 = Solution(
            np.array([4.1, 0.9], dtype=np.float64), metadata="metadata6"
        )

        self.pareto_front.add_solution(solution_1)
        self.pareto_front.add_solution(solution_2)
        self.pareto_front.add_solution(solution_3)
        self.pareto_front.add_solution(solution_4)
        self.pareto_front.add_solution(solution_5)
        self.pareto_front.add_solution(solution_6)

        representative_solutions = self.pareto_front.select_representative_solutions(
            num_solutions=2
        )
        self.assertEqual(len(representative_solutions), 2)
        self.assertTrue(
            any(
                np.array_equal(solution_2.get_values(), p.get_values())
                for p in representative_solutions
            )
        )
        self.assertTrue(
            any(
                np.array_equal(solution_5.get_values(), p.get_values())
                for p in representative_solutions
            )
        )


class TestGeneWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.image_matrix = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        self.generator = GeneWindowGenerator(
            image_matrix=self.image_matrix,
            width_ratio=1,
            height_ratio=1,
            min_population_size=1,
            random_seed=42,
            fitness_function_evaluations=50,
        )

    def test_is_window_generator_interface(self):
        self.assertTrue(issubclass(GeneWindowGenerator, IWindowGenerator))

    def test_generate_random_rectangle(self):
        rect = self.generator.generate_random_rectangle()
        self.assertTrue(0 <= rect[0] < self.generator.width)
        self.assertTrue(0 <= rect[1] < self.generator.height)
        self.assertTrue(0 < rect[2] <= self.generator.width)
        self.assertTrue(0 < rect[3] <= self.generator.height)
        self.assertTrue(rect[0] + rect[2] <= self.generator.width)
        self.assertTrue(rect[1] + rect[3] <= self.generator.height)

    def test_generate_random_window(self):
        window = self.generator.generate_random_window()
        self.assertIsInstance(window, Window)
        self.assertTrue(0 <= window.i < self.generator.width)
        self.assertTrue(0 <= window.j < self.generator.height)
        self.assertTrue(0 < window.window_width <= self.generator.width)
        self.assertTrue(0 < window.window_height <= self.generator.height)
        self.assertTrue(window.i + window.window_width <= self.generator.width)
        self.assertTrue(window.j + window.window_height <= self.generator.height)

    def test_crossover(self):
        parent1 = self.generator.generate_random_window()
        parent2 = self.generator.generate_random_window()

        # random window
        self.generator.population = []
        child_window = self.generator.crossover(parent1, parent2)
        self.assertTrue(0 <= child_window.i < self.generator.width)
        self.assertTrue(0 <= child_window.j < self.generator.height)
        self.assertTrue(0 < child_window.window_width <= self.generator.width)
        self.assertTrue(0 < child_window.window_height <= self.generator.height)
        self.assertTrue(
            child_window.i + child_window.window_width <= self.generator.width
        )
        self.assertTrue(
            child_window.j + child_window.window_height <= self.generator.height
        )

        # real crossover
        self.generator.population = [parent1, parent2]
        child_window = self.generator.crossover(parent1, parent2)
        self.assertTrue(
            parent1.i <= child_window.i <= parent2.i
            or parent2.i <= child_window.i <= parent1.i
        )
        self.assertTrue(
            parent1.j <= child_window.j <= parent2.j
            or parent2.j <= child_window.j <= parent1.j
        )
        self.assertTrue(
            parent1.window_width <= child_window.window_width <= parent2.window_width
            or parent2.window_width <= child_window.window_width <= parent1.window_width
        )
        self.assertTrue(
            parent1.window_height <= child_window.window_height <= parent2.window_height
            or parent2.window_height
            <= child_window.window_height
            <= parent1.window_height
        )

    def test_mutation(self):
        parent_window = self.generator.generate_random_window()

        # random window
        self.generator.population = []
        child_window = self.generator.mutation(parent_window)
        self.assertTrue(0 <= child_window.i < self.generator.width)
        self.assertTrue(0 <= child_window.j < self.generator.height)
        self.assertTrue(0 < child_window.window_width <= self.generator.width)
        self.assertTrue(0 < child_window.window_height <= self.generator.height)
        self.assertTrue(
            child_window.i + child_window.window_width <= self.generator.width
        )
        self.assertTrue(
            child_window.j + child_window.window_height <= self.generator.height
        )

        # real mutation
        self.generator.population = [parent_window]
        mutated_window = self.generator.mutation(parent_window)
        self.assertIsInstance(mutated_window, Window)
        self.assertTrue(0 <= mutated_window.i < self.generator.width)
        self.assertTrue(0 <= mutated_window.j < self.generator.height)
        self.assertTrue(0 < mutated_window.window_width <= self.generator.width)
        self.assertTrue(0 < mutated_window.window_height <= self.generator.height)
        self.assertTrue(
            mutated_window.i + mutated_window.window_width <= self.generator.width
        )
        self.assertTrue(
            mutated_window.j + mutated_window.window_height <= self.generator.height
        )

    def test_generate_windows(self):
        windows = list(self.generator.generate_windows())

        self.assertEqual(len(windows), self.generator.fitness_function_evaluations)
        for window in windows:
            self.assertIsInstance(window, Window)
            self.assertTrue(0 <= window.i < self.generator.width)
            self.assertTrue(0 <= window.j < self.generator.height)
            self.assertTrue(0 < window.window_width <= self.generator.width)
            self.assertTrue(0 < window.window_height <= self.generator.height)
            self.assertTrue(window.i + window.window_width <= self.generator.width)
            self.assertTrue(window.j + window.window_height <= self.generator.height)


if __name__ == "__main__":
    unittest.main()
