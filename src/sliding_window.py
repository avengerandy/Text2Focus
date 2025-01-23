from dataclasses import dataclass
import numpy as np
from typing import Generator


@dataclass
class Stride:
    horizontal: int
    vertical: int

    def set_stride(self, vertical: int, horizontal: int):
        self.horizontal = max(horizontal, 1)
        self.vertical = max(vertical, 1)


@dataclass(frozen=True)
class Increment:
    width: int
    height: int


@dataclass
class Shape:
    width: int
    height: int

    def expand(self, increment: Increment):
        self.width += max(increment.width, 1)
        self.height += max(increment.height, 1)


@dataclass(frozen=True)
class Window:
    sub_array : np.ndarray
    i : int
    j : int
    window_width : int
    window_height : int


class SlidingWindowScanner:
    def __init__(self, arr: np.ndarray, shape: Shape, stride: Stride):
        self.arr = arr
        self.shape = shape
        self.stride = stride

    def sliding_window_scan(self) -> Generator[np.ndarray, None, None]:
        height, width = self.arr.shape
        window_width, window_height = self.shape.width, self.shape.height
        horizontal_stride, vertical_stride = self.stride.horizontal, self.stride.vertical

        for i in range(0, width - window_width + 1, horizontal_stride):
            for j in range(0, height - window_height + 1, vertical_stride):
                sub_array = self.arr[j:j + window_height, i:i + window_width]
                yield Window(sub_array=sub_array, i=i, j=j, window_width=window_width, window_height=window_height)


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
