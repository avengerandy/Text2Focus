from dataclasses import dataclass
import numpy as np
from typing import Generator


@dataclass
class Stride:
    vertical: int
    horizontal: int

    def set_stride(self, vertical: int, horizontal: int):
        self.vertical = max(vertical, 1)
        self.horizontal = max(horizontal, 1)


@dataclass(frozen=True)
class Increment:
    height: int
    width: int


@dataclass
class Shape:
    height: int
    width: int

    def expand(self, increment: Increment):
        self.height += max(increment.height, 1)
        self.width += max(increment.width, 1)


class SlidingWindowScanner:
    def __init__(self, arr: np.ndarray, shape: Shape, stride: Stride):
        self.arr = arr
        self.shape = shape
        self.stride = stride

    def sliding_window_scan(self) -> Generator[np.ndarray, None, None]:
        height, width = self.arr.shape
        window_height, window_width = self.shape.height, self.shape.width
        vertical_stride, horizontal_stride = self.stride.vertical, self.stride.horizontal

        for i in range(0, height - window_height + 1, vertical_stride):
            for j in range(0, width - window_width + 1, horizontal_stride):
                sub_array = self.arr[i:i + window_height, j:j + window_width]
                yield sub_array


class SlidingWindowProcessor:
    def __init__(self, arr: np.ndarray, shape: Shape, stride: Stride, increment: Increment):
        self.arr = arr
        self.shape = shape
        self.stride = stride
        self.increment = increment

    def process(self) -> Generator[np.ndarray, None, None]:
        scanner = SlidingWindowScanner(self.arr, self.shape, self.stride)

        while self.shape.height <= self.arr.shape[0] and self.shape.width <= self.arr.shape[1]:
            for sub_array in scanner.sliding_window_scan():
                yield sub_array

            self.shape.expand(self.increment)
