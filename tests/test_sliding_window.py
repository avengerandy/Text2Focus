import unittest
import numpy as np
from dataclasses import FrozenInstanceError
from src.sliding_window import Stride, Increment, Shape, Window, SlidingWindowScanner, SlidingWindowProcessor


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

    def test_increment_immutable(self):
        increment = Increment(1, 1)

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


class TestWindow(unittest.TestCase):

    def test_window_creation(self):
        sub_array = np.array([[1, 2], [3, 4]])
        i, j = 0, 0
        window_height, window_width = 2, 2

        window = Window(sub_array=sub_array, i=i, j=j, window_height=window_height, window_width=window_width)

        np.testing.assert_array_equal(window.sub_array, sub_array)
        self.assertEqual(window.i, i)
        self.assertEqual(window.j, j)
        self.assertEqual(window.window_height, window_height)
        self.assertEqual(window.window_width, window_width)

    def test_window_immutable(self):
        window = Window(sub_array=np.array([[1, 2], [3, 4]]), i=0, j=0, window_height=2, window_width=2)

        with self.assertRaises(FrozenInstanceError):
            window.sub_array = np.array([[5, 6], [7, 8]])

        with self.assertRaises(FrozenInstanceError):
            window.i = 1

        with self.assertRaises(FrozenInstanceError):
            window.j = 1

        with self.assertRaises(FrozenInstanceError):
            window.window_height = 3

        with self.assertRaises(FrozenInstanceError):
            window.window_width = 3

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
            Window(sub_array=np.array([[1, 2], [5, 6]]), i=0, j=0, window_height=2, window_width=2),
            Window(sub_array=np.array([[2, 3], [6, 7]]), i=0, j=1, window_height=2, window_width=2),
            Window(sub_array=np.array([[3, 4], [7, 8]]), i=0, j=2, window_height=2, window_width=2),
            Window(sub_array=np.array([[5, 6], [9, 10]]), i=1, j=0, window_height=2, window_width=2),
            Window(sub_array=np.array([[6, 7], [10, 11]]), i=1, j=1, window_height=2, window_width=2),
            Window(sub_array=np.array([[7, 8], [11, 12]]), i=1, j=2, window_height=2, window_width=2),
            Window(sub_array=np.array([[9, 10], [13, 14]]), i=2, j=0, window_height=2, window_width=2),
            Window(sub_array=np.array([[10, 11], [14, 15]]), i=2, j=1, window_height=2, window_width=2),
            Window(sub_array=np.array([[11, 12], [15, 16]]), i=2, j=2, window_height=2, window_width=2)
        ]

        results = list(scanner.sliding_window_scan())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected.sub_array, result.sub_array)
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)

    def test_sliding_window_scan_out_of_bounds(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
        shape = Shape(2, 2)
        stride = Stride(3, 3)

        scanner = SlidingWindowScanner(arr, shape, stride)

        expected_results = [
            Window(sub_array=np.array([[1, 2], [5, 6]]), i=0, j=0, window_height=2, window_width=2)
        ]

        results = list(scanner.sliding_window_scan())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected.sub_array, result.sub_array)
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)


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
            Window(sub_array=np.array([[1, 2], [5, 6]]), i=0, j=0, window_height=2, window_width=2),
            Window(sub_array=np.array([[2, 3], [6, 7]]), i=0, j=1, window_height=2, window_width=2),
            Window(sub_array=np.array([[3, 4], [7, 8]]), i=0, j=2, window_height=2, window_width=2),
            Window(sub_array=np.array([[5, 6], [9, 10]]), i=1, j=0, window_height=2, window_width=2),
            Window(sub_array=np.array([[6, 7], [10, 11]]), i=1, j=1, window_height=2, window_width=2),
            Window(sub_array=np.array([[7, 8], [11, 12]]), i=1, j=2, window_height=2, window_width=2),
            Window(sub_array=np.array([[9, 10], [13, 14]]), i=2, j=0, window_height=2, window_width=2),
            Window(sub_array=np.array([[10, 11], [14, 15]]), i=2, j=1, window_height=2, window_width=2),
            Window(sub_array=np.array([[11, 12], [15, 16]]), i=2, j=2, window_height=2, window_width=2),

            Window(sub_array=np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]]), i=0, j=0, window_height=3, window_width=3),
            Window(sub_array=np.array([[2, 3, 4], [6, 7, 8], [10, 11, 12]]), i=0, j=1, window_height=3, window_width=3),
            Window(sub_array=np.array([[5, 6, 7], [9, 10, 11], [13, 14, 15]]), i=1, j=0, window_height=3, window_width=3),
            Window(sub_array=np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16]]), i=1, j=1, window_height=3, window_width=3),
        ]

        results = list(processor.process())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected.sub_array, result.sub_array)
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)

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
            Window(sub_array=np.array([[1, 2], [5, 6]]), i=0, j=0, window_height=2, window_width=2)
        ]

        results = list(processor.process())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(expected.sub_array, result.sub_array)
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)


if __name__ == "__main__":
    unittest.main()
