import unittest
import numpy as np
from dataclasses import FrozenInstanceError
from src.sliding_window import Stride, Increment, Shape, SlidingWindowScanner, SlidingWindowProcessor


class TestStride(unittest.TestCase):

    def test_set_stride(self):
        stride = Stride(1, 1)
        stride.set_stride(2, 2)

        self.assertEqual(stride.vertical, 2)
        self.assertEqual(stride.horizontal, 2)


class TestIncrement(unittest.TestCase):

    def test_increment_creation(self):
        increment = Increment(1, 1)

        self.assertEqual(increment.height, 1)
        self.assertEqual(increment.width, 1)

        with self.assertRaises(FrozenInstanceError):
            increment.height = 2
        with self.assertRaises(FrozenInstanceError):
            increment.width = 2


class TestShape(unittest.TestCase):

    def test_expand(self):
        shape = Shape(3, 3)

        increment = Increment(1, 1)
        shape.expand(increment)

        self.assertEqual(shape.height, 4)
        self.assertEqual(shape.width, 4)


class TestSlidingWindowScanner(unittest.TestCase):

    def test_sliding_window_scan(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
        shape = Shape(2, 2)
        stride = Stride(1, 1)

        scanner = SlidingWindowScanner(arr, shape, stride)

        expected_results = [
            np.array([[1, 2], [5, 6]]),
            np.array([[2, 3], [6, 7]]),
            np.array([[3, 4], [7, 8]]),
            np.array([[5, 6], [9, 10]]),
            np.array([[6, 7], [10, 11]]),
            np.array([[7, 8], [11, 12]]),
            np.array([[9, 10], [13, 14]]),
            np.array([[10, 11], [14, 15]]),
            np.array([[11, 12], [15, 16]])
        ]

        results = list(scanner.sliding_window_scan())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected, result)

    def test_sliding_window_scan_out_of_bounds(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
        shape = Shape(2, 2)
        stride = Stride(3, 3)

        scanner = SlidingWindowScanner(arr, shape, stride)

        expected_results = [
            np.array([[1, 2], [5, 6]])
        ]

        results = list(scanner.sliding_window_scan())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected, result)


class TestSlidingWindowProcessor(unittest.TestCase):

    def test_process(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
        shape = Shape(2, 2)
        stride = Stride(1, 1)
        increment = Increment(1, 1)

        processor = SlidingWindowProcessor(arr, shape, stride, increment)

        expected_results = [
            np.array([[1, 2], [5, 6]]),
            np.array([[2, 3], [6, 7]]),
            np.array([[3, 4], [7, 8]]),
            np.array([[5, 6], [9, 10]]),
            np.array([[6, 7], [10, 11]]),
            np.array([[7, 8], [11, 12]]),
            np.array([[9, 10], [13, 14]]),
            np.array([[10, 11], [14, 15]]),
            np.array([[11, 12], [15, 16]]),

            np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]]),
            np.array([[2, 3, 4], [6, 7, 8], [10, 11, 12]]),
            np.array([[5, 6, 7], [9, 10, 11], [13, 14, 15]]),
            np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16]]),
        ]

        results = list(processor.process())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected, result)

    def test_process_out_of_bounds(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
        shape = Shape(2, 2)
        stride = Stride(1, 1)
        increment = Increment(3, 3)
        processor = SlidingWindowProcessor(arr, shape, stride, increment)

        expected_results = [
            np.array([[1, 2], [5, 6]])
        ]

        results = list(processor.process())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
