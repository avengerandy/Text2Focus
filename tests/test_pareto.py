import unittest
import numpy as np
from src.pareto import Solution, DynamicRowMatrix, ParetoFront

class TestSolution(unittest.TestCase):

    def test_is_dominated_by(self):
        sol1 = Solution(np.array([1, 2, 3]))
        sol2 = Solution(np.array([3, 2, 1]))
        sol3 = Solution(np.array([3, 4, 5]))

        self.assertEqual(sol1.is_dominated_by(sol2), np.bool_(False))
        self.assertEqual(sol2.is_dominated_by(sol1), np.bool_(False))
        self.assertEqual(sol1.is_dominated_by(sol3), np.bool_(True))

        self.assertEqual(sol2.is_dominated_by(sol3), np.bool_(True))
        self.assertEqual(sol3.is_dominated_by(sol2), np.bool_(False))

    def test_get_values(self):
        sol = Solution(np.array([1, 2, 3]))
        self.assertTrue(np.array_equal(sol.get_values(), np.array([1, 2, 3])))

    def test_repr(self):
        sol = Solution(np.array([1, 2, 3]))
        self.assertEqual(repr(sol), "Solution([1 2 3])")

        sol = Solution(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.assertEqual(repr(sol), "Solution([1 2 3 4 5]...)")

    def test_edge_case_empty(self):
        sol1 = Solution(np.array([]))
        sol2 = Solution(np.array([]))

        self.assertFalse(sol1.is_dominated_by(sol2))
        self.assertFalse(sol2.is_dominated_by(sol1))

    def test_edge_case_single_element(self):
        sol1 = Solution(np.array([1]))
        sol2 = Solution(np.array([2]))

        self.assertTrue(sol1.is_dominated_by(sol2))
        self.assertFalse(sol2.is_dominated_by(sol1))


class TestDynamicRowMatrix(unittest.TestCase):

    def setUp(self):
        self.matrix = DynamicRowMatrix(initial_column_capacity=3, initial_row_capacity=2)

    def test_initial_size_and_capacity(self):
        self.assertEqual(self.matrix.get_size(), 0)
        self.assertEqual(self.matrix._row_capacity, 2)
        self.assertEqual(self.matrix._column_capacity, 3)

    def test_add_row(self):
        new_row = np.array([1.0, 2.0, 3.0])
        self.matrix.add_row(new_row)
        self.assertEqual(self.matrix.get_size(), 1)
        np.testing.assert_array_equal(self.matrix.get_data()[0], new_row)

    def test_expand_capacity(self):
        for i in range(3):
            self.matrix.add_row(np.array([1.0, 2.0, 3.0]))

        self.assertEqual(self.matrix.get_size(), 3)
        self.assertGreater(self.matrix._row_capacity, 2)

    def test_capacity_does_not_exceed_when_not_needed(self):
        initial_capacity = self.matrix._row_capacity
        self.matrix.add_row(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(self.matrix.get_size(), 1)
        self.assertEqual(self.matrix._row_capacity, initial_capacity)

    def test_get_data(self):
        new_row1 = np.array([1.0, 2.0, 3.0])
        new_row2 = np.array([4.0, 5.0, 6.0])

        self.matrix.add_row(new_row1)
        self.matrix.add_row(new_row2)

        data = self.matrix.get_data()
        self.assertEqual(data.shape, (2, 3))
        np.testing.assert_array_equal(data[0], new_row1)
        np.testing.assert_array_equal(data[1], new_row2)

    def test_value_error_on_invalid_data(self):
        invalid_row = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            self.matrix.add_row(invalid_row)


class TestParetoFront(unittest.TestCase):

    def setUp(self):
        self.pareto_front = ParetoFront(solution_dimensions=2)

    def test_add_solution(self):
        solution_1 = Solution(np.array([1, 2], dtype=np.float64))
        solution_2 = Solution(np.array([2, 1], dtype=np.float64))

        self.pareto_front.add_solution(solution_1)
        self.pareto_front.add_solution(solution_2)

        pareto_solutions = self.pareto_front.get_pareto_solutions()
        self.assertEqual(len(pareto_solutions), 2)

    def test_get_pareto_solutions(self):
        solution_1 = Solution(np.array([1, 2], dtype=np.float64))
        solution_2 = Solution(np.array([2, 1], dtype=np.float64))
        solution_3 = Solution(np.array([3, 1], dtype=np.float64))

        self.pareto_front.add_solution(solution_1)
        self.pareto_front.add_solution(solution_2)
        self.pareto_front.add_solution(solution_3)

        pareto_solutions = self.pareto_front.get_pareto_solutions()

        self.assertEqual(len(pareto_solutions), 2)
        self.assertTrue(any(np.array_equal(solution_1.get_values(), p.get_values()) for p in pareto_solutions))
        self.assertTrue(any(np.array_equal(solution_3.get_values(), p.get_values()) for p in pareto_solutions))
        self.assertFalse(any(np.array_equal(solution_2.get_values(), p.get_values()) for p in pareto_solutions))

    def test_get_elbow_point(self):
        solution_1 = Solution(np.array([10, 9], dtype=np.float64))
        solution_2 = Solution(np.array([20, 8], dtype=np.float64))
        solution_3 = Solution(np.array([40, 1], dtype=np.float64))
        solution_4 = Solution(np.array([50, 2], dtype=np.float64))

        self.pareto_front.add_solution(solution_1)
        self.pareto_front.add_solution(solution_2)
        self.pareto_front.add_solution(solution_3)
        self.pareto_front.add_solution(solution_4)

        elbow_point = self.pareto_front.get_elbow_point()
        self.assertTrue(np.array_equal(solution_2.get_values(), elbow_point.get_values()))

    def test_get_point_by_weight(self):
        solution_1 = Solution(np.array([1, 3], dtype=np.float64))
        solution_2 = Solution(np.array([2, 2], dtype=np.float64))
        solution_3 = Solution(np.array([5, 1], dtype=np.float64))

        self.pareto_front.add_solution(solution_1)
        self.pareto_front.add_solution(solution_2)
        self.pareto_front.add_solution(solution_3)

        weight = [0.5, 0.5]
        point = self.pareto_front.get_point_by_weight(weight)
        self.assertTrue(np.array_equal(solution_3.get_values(), point.get_values()))

        weight = [0.1, 0.9]
        point = self.pareto_front.get_point_by_weight(weight)
        self.assertTrue(np.array_equal(solution_1.get_values(), point.get_values()))

if __name__ == '__main__':
    unittest.main()
