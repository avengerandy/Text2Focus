import unittest
import numpy as np
from src.fitness import total_sum, total_positive_ratio

class TestTotalFunctions(unittest.TestCase):

    def test_total_sum(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(total_sum(data), 21)

        data = np.array([[-1, -2], [3, 4]])
        self.assertEqual(total_sum(data), 4)

        data = np.array([[0, 0], [0, 0]])
        self.assertEqual(total_sum(data), 0)

    def test_total_positive_ratio(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(total_positive_ratio(data), 1.0)

        data = np.array([[-1, -2], [3, 4]])
        self.assertEqual(total_positive_ratio(data), 0.5)

        data = np.array([[0, 0], [0, 0]])
        self.assertEqual(total_positive_ratio(data), 0.0)

        data = np.array([[0.0001, 0.0002], [0.0003, 0.0004]])
        self.assertEqual(total_positive_ratio(data, epsilon=0.001), 0.0)

        data = np.array([[0.0001, 0.0002], [0.0003, 0.01]])
        self.assertEqual(total_positive_ratio(data, epsilon=0.001), 0.25)


if __name__ == "__main__":
    unittest.main()
