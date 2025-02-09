import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from src.sliding_window import (
    Increment,
    Shape,
    SlidingWindowProcessor,
    SlidingWindowScanner,
    Stride,
    Window,
)


class TestStride(unittest.TestCase):

    def test_creation(self):
        stride = Stride(horizontal=1, vertical=2)

        self.assertEqual(stride.horizontal, 1)
        self.assertEqual(stride.vertical, 2)

    def test_immutable(self):
        stride = Stride(horizontal=1, vertical=1)

        with self.assertRaises(FrozenInstanceError):
            stride.vertical = 2
        with self.assertRaises(FrozenInstanceError):
            stride.horizontal = 2


class TestIncrement(unittest.TestCase):

    def test_creation(self):
        increment = Increment(width=1, height=2)

        self.assertEqual(increment.width, 1)
        self.assertEqual(increment.height, 2)

    def test_immutable(self):
        increment = Increment(width=1, height=1)

        with self.assertRaises(FrozenInstanceError):
            increment.height = 2
        with self.assertRaises(FrozenInstanceError):
            increment.width = 2


class TestShape(unittest.TestCase):

    def test_expand(self):
        shape = Shape(width=2, height=3)

        increment = Increment(width=1, height=2)
        shape.expand(increment)

        self.assertEqual(shape.width, 3)
        self.assertEqual(shape.height, 5)


class TestWindow(unittest.TestCase):

    def test_window_creation(self):
        sub_image_matrix = np.array([[1, 2], [3, 4]])
        i, j = 0, 0
        window_height, window_width = 2, 2

        window = Window(
            sub_image_matrix=sub_image_matrix,
            i=i,
            j=j,
            window_height=window_height,
            window_width=window_width,
        )

        np.testing.assert_array_equal(window.sub_image_matrix, sub_image_matrix)
        self.assertEqual(window.i, i)
        self.assertEqual(window.j, j)
        self.assertEqual(window.window_height, window_height)
        self.assertEqual(window.window_width, window_width)

    def test_window_immutable(self):
        window = Window(
            sub_image_matrix=np.array([[1, 2], [3, 4]]),
            i=0,
            j=0,
            window_height=2,
            window_width=2,
        )

        with self.assertRaises(FrozenInstanceError):
            window.sub_image_matrix = np.array([[5, 6], [7, 8]])

        with self.assertRaises(FrozenInstanceError):
            window.i = 1

        with self.assertRaises(FrozenInstanceError):
            window.j = 1

        with self.assertRaises(FrozenInstanceError):
            window.window_height = 3

        with self.assertRaises(FrozenInstanceError):
            window.window_width = 3


class TestSlidingWindowScanner(unittest.TestCase):

    def test_generate_windows(self):
        image_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        shape = Shape(width=2, height=2)
        stride = Stride(horizontal=1, vertical=1)

        scanner = SlidingWindowScanner(image_matrix, shape, stride)

        expected_results = [
            Window(
                sub_image_matrix=np.array([[1, 2], [5, 6]]),
                i=0,
                j=0,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[5, 6], [9, 10]]),
                i=0,
                j=1,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[9, 10], [13, 14]]),
                i=0,
                j=2,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[2, 3], [6, 7]]),
                i=1,
                j=0,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[6, 7], [10, 11]]),
                i=1,
                j=1,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[10, 11], [14, 15]]),
                i=1,
                j=2,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[3, 4], [7, 8]]),
                i=2,
                j=0,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[7, 8], [11, 12]]),
                i=2,
                j=1,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[11, 12], [15, 16]]),
                i=2,
                j=2,
                window_height=2,
                window_width=2,
            ),
        ]

        results = list(scanner.generate_windows())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(
                expected.sub_image_matrix, result.sub_image_matrix
            )
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)

    def test_generate_windows_out_of_bounds(self):
        image_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        shape = Shape(width=2, height=2)
        stride = Stride(horizontal=3, vertical=3)

        scanner = SlidingWindowScanner(image_matrix, shape, stride)

        expected_results = [
            Window(
                sub_image_matrix=np.array([[1, 2], [5, 6]]),
                i=0,
                j=0,
                window_height=2,
                window_width=2,
            )
        ]

        results = list(scanner.generate_windows())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(
                expected.sub_image_matrix, result.sub_image_matrix
            )
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)


class TestSlidingWindowProcessor(unittest.TestCase):

    def test_process(self):
        image_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        shape = Shape(width=2, height=2)
        stride = Stride(horizontal=1, vertical=1)
        increment = Increment(width=1, height=1)
        scanner = SlidingWindowScanner(image_matrix, shape, stride)
        processor = SlidingWindowProcessor(scanner, increment)

        expected_results = [
            Window(
                sub_image_matrix=np.array([[1, 2], [5, 6]]),
                i=0,
                j=0,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[5, 6], [9, 10]]),
                i=0,
                j=1,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[9, 10], [13, 14]]),
                i=0,
                j=2,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[2, 3], [6, 7]]),
                i=1,
                j=0,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[6, 7], [10, 11]]),
                i=1,
                j=1,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[10, 11], [14, 15]]),
                i=1,
                j=2,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[3, 4], [7, 8]]),
                i=2,
                j=0,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[7, 8], [11, 12]]),
                i=2,
                j=1,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[11, 12], [15, 16]]),
                i=2,
                j=2,
                window_height=2,
                window_width=2,
            ),
            Window(
                sub_image_matrix=np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]]),
                i=0,
                j=0,
                window_height=3,
                window_width=3,
            ),
            Window(
                sub_image_matrix=np.array([[5, 6, 7], [9, 10, 11], [13, 14, 15]]),
                i=0,
                j=1,
                window_height=3,
                window_width=3,
            ),
            Window(
                sub_image_matrix=np.array([[2, 3, 4], [6, 7, 8], [10, 11, 12]]),
                i=1,
                j=0,
                window_height=3,
                window_width=3,
            ),
            Window(
                sub_image_matrix=np.array([[6, 7, 8], [10, 11, 12], [14, 15, 16]]),
                i=1,
                j=1,
                window_height=3,
                window_width=3,
            ),
        ]

        results = list(processor.generate_windows())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(
                expected.sub_image_matrix, result.sub_image_matrix
            )
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)

    def test_process_out_of_bounds(self):
        image_matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        shape = Shape(width=2, height=2)
        stride = Stride(horizontal=1, vertical=1)
        increment = Increment(width=3, height=3)
        scanner = SlidingWindowScanner(image_matrix, shape, stride)
        processor = SlidingWindowProcessor(scanner, increment)

        expected_results = [
            Window(
                sub_image_matrix=np.array([[1, 2], [5, 6]]),
                i=0,
                j=0,
                window_height=2,
                window_width=2,
            )
        ]

        results = list(processor.generate_windows())
        for expected, result in zip(expected_results, results):
            np.testing.assert_array_equal(
                expected.sub_image_matrix, result.sub_image_matrix
            )
            self.assertEqual(expected.i, result.i)
            self.assertEqual(expected.j, result.j)
            self.assertEqual(expected.window_height, result.window_height)
            self.assertEqual(expected.window_width, result.window_width)


if __name__ == "__main__":
    unittest.main()
