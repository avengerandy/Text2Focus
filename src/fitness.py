"""
This module contains a set of fitness functions used to evaluate the quality
of an image matrix.
"""

import numpy as np


def image_matrix_sum(image_matrix: np.ndarray) -> float:
    """
    Calculate the sum of all elements in the image matrix.

    Parameters:
        image_matrix (np.ndarray): The image matrix to evaluate.

    Returns:
        float: The sum of all elements in the image matrix.
    """
    return np.sum(image_matrix)


def image_matrix_average(image_matrix: np.ndarray) -> float:
    """
    Calculate the average value of all elements in the image matrix.

    Parameters:
        image_matrix (np.ndarray): The image matrix to evaluate.

    Returns:
        float: The average value of all elements in the image matrix.
    """
    return np.sum(image_matrix) / image_matrix.size


def image_matrix_negative_boundary_average(image_matrix: np.ndarray) -> float:
    """
    Calculate the average value of the boundary elements of the image matrix.

    Parameters:
        image_matrix (np.ndarray): The image matrix to evaluate.

    Returns:
        float: The average value of the boundary elements of the image matrix.
    """
    top_row = image_matrix[0, :]
    bottom_row = image_matrix[-1, :]
    left_column = image_matrix[1:-1, 0]
    right_column = image_matrix[1:-1, -1]

    boundary_sum = (
        np.sum(top_row)
        + np.sum(bottom_row)
        + np.sum(left_column)
        + np.sum(right_column)
    )
    boundary_count = (
        top_row.size + bottom_row.size + left_column.size + right_column.size
    )

    return -boundary_sum / boundary_count
