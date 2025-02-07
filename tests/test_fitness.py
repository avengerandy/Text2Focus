import unittest

import numpy as np

from src.fitness import (
    image_matrix_average,
    image_matrix_negative_boundary_average,
    image_matrix_sum,
)


class TestTotalFunctions(unittest.TestCase):

    def test_image_matrix_sum(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(image_matrix_sum(data), 21)

        data = np.array([[-1, -2], [3, 4]])
        self.assertEqual(image_matrix_sum(data), 4)

        data = np.array([[0, 0], [0, 0]])
        self.assertEqual(image_matrix_sum(data), 0)

    def test_image_matrix_average(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(image_matrix_average(data), 3.5)

        data = np.array([[-1, -2], [3, 4]])
        self.assertEqual(image_matrix_average(data), 1.0)

        data = np.array([[0, 0], [0, 0]])
        self.assertEqual(image_matrix_average(data), 0.0)

    def test_image_matrix_negative_boundary_average(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(image_matrix_negative_boundary_average(data), -5.0)

        data = np.array([[-1, -2], [3, 4]])
        self.assertEqual(image_matrix_negative_boundary_average(data), -1.0)

        data = np.array([[0, 0], [0, 0]])
        self.assertEqual(image_matrix_negative_boundary_average(data), -0.0)


if __name__ == "__main__":
    unittest.main()
