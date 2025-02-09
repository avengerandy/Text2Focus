"""
This module provides classes and methods for performing sliding window
operations on an image matrix.

Classes:
    - Stride: Defines the horizontal and vertical steps (strides) for sliding
      the window.
    - Increment: Defines the amount by which the window's dimensions (width
      and height) should be increased.
    - Shape: Defines the initial dimensions of the sliding window and provides
      a method to expand its size.
    - Window: Represents a single sliding window over the image matrix.
    - SlidingWindowScanner: Responsible for generating sliding windows over
      the image matrix using the given window shape and stride.
    - SlidingWindowProcessor: Coordinates the process of sliding windows over
      the image matrix and dynamically expanding the window's shape.

Usage:
    - You can create a SlidingWindowProcessor with a specific image matrix,
      window shape, stride, and increment to perform the sliding window
      operation.
    - The processor will yield individual windows (sub-matrices) that can be
      used for further processing.
"""

from dataclasses import dataclass
from typing import Generator
from abc import ABC, abstractmethod

import numpy as np


@dataclass(frozen=True)
class Stride:
    """
    Represents the stride (step size) for moving the sliding window.

    Attributes:
        horizontal (int): The horizontal step size for sliding the window.
        vertical (int): The vertical step size for sliding the window.
    """

    horizontal: int
    vertical: int


@dataclass(frozen=True)
class Increment:
    """
    Represents the increment in the dimensions (width and height) of the window.

    Attributes:
        width (int): The increment in width of the window after each iteration.
        height (int): The increment in height of the window after each iteration.
    """

    width: int
    height: int


@dataclass
class Shape:
    """
    Represents the shape (dimensions) of the sliding window.

    Attributes:
        width (int): The width of the window.
        height (int): The height of the window.
    """

    width: int
    height: int

    def expand(self, increment: Increment):
        """
        Expands the dimensions of the window by the specified increment.

        This method updates the width and height of the window by adding the
        corresponding increment values. The minimum increment for each dimension
        is 1 to avoid non-positive window sizes.

        Args:
            increment (Increment): The increment in width and height to expand
            the window by.
        """
        self.width += max(increment.width, 1)
        self.height += max(increment.height, 1)


@dataclass(frozen=True)
class Window:
    """
    Represents a single sliding window over the image matrix.

    Attributes:
        sub_image_matrix (np.ndarray): The portion of the image matrix
        corresponding to this window.
        i (int): The starting horizontal index of the window.
        j (int): The starting vertical index of the window.
        window_width (int): The width of the window.
        window_height (int): The height of the window.
    """

    sub_image_matrix: np.ndarray
    i: int
    j: int
    window_width: int
    window_height: int


class IWindowGenerator(ABC):
    """
    Interface for a window generator.
    """

    @abstractmethod
    def generate_windows(self) -> Generator[Window, None, None]:
        """
        Generate Window objects over the image matrix.

        Yields:
            Window: The window object containing the sub-matrix, window position,
            and dimensions.
        """


class SlidingWindowScanner(IWindowGenerator):
    """
    Scans the image matrix using a sliding window approach.

    This class generates sliding windows over the image matrix based on the
    provided window shape and stride.

    Attributes:
        image_matrix (np.ndarray): The image matrix (2D array) to scan.
        shape (Shape): The shape (width and height) of the sliding window.
        stride (Stride): The stride (step size) for moving the window horizontally
        and vertically.
    """

    def __init__(self, image_matrix: np.ndarray, shape: Shape, stride: Stride):
        self.image_matrix = image_matrix
        self.shape = shape
        self.stride = stride

    def generate_windows(self) -> Generator[Window, None, None]:
        """
        Generates sliding windows over the image matrix based on the given
        window shape and stride.

        This method yields `Window` objects as it slides the window across the
        image matrix.

        Yields:
            Window: The window object containing the sub-matrix, window position,
            and dimensions.
        """
        height, width = self.image_matrix.shape
        window_width, window_height = self.shape.width, self.shape.height
        horizontal_stride, vertical_stride = (
            self.stride.horizontal,
            self.stride.vertical,
        )

        # scan (move horizontally and vertically by stride amounts)
        for i in range(0, width - window_width + 1, horizontal_stride):
            for j in range(0, height - window_height + 1, vertical_stride):
                # Extract the sub-matrix for this window
                sub_image_matrix = self.image_matrix[
                    j : j + window_height, i : i + window_width
                ]
                yield Window(
                    sub_image_matrix=sub_image_matrix,
                    i=i,
                    j=j,
                    window_width=window_width,
                    window_height=window_height,
                )


class SlidingWindowProcessor(IWindowGenerator):
    """
    Processes an image matrix using sliding windows and adjusts the window size
    dynamically.

    This class manages the sliding window operation and increases the window size
    after each scan by the given increment.

    Attributes:
        scanner (SlidingWindowScanner): The scanner object that generates sliding
        windows over the image matrix.
        increment (Increment): The amount by which the window size is incremented
        after each scan.
    """

    def __init__(
        self,
        scanner: SlidingWindowScanner,
        increment: Increment,
    ):
        self.scanner = scanner
        self.increment = increment

    def generate_windows(self) -> Generator[Window, None, None]:
        """
        Processes the image matrix with a sliding window and dynamically increases
        the window size.

        This method repeatedly generates windows using the `SlidingWindowScanner`,
        expanding the window after each complete scan.

        Yields:
            Window: The window object containing the sub-matrix, window position,
            and dimensions.
        """
        while (
            self.scanner.shape.height <= self.scanner.image_matrix.shape[0]
            and self.scanner.shape.width <= self.scanner.image_matrix.shape[1]
        ):
            yield from self.scanner.generate_windows()
            self.scanner.shape.expand(self.increment)
