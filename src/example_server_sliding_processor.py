"""
This module is a example of how to use

* CoordinateTransformer (resize image)
* SlidingWindowProcessor (brute force)
* DividedParetoFront (divided and conquer)

to find the best crop for a given image.
"""

# pylint: disable=R0801
# R0801: duplicate-code

import numpy as np

from src.accelerator import DividedParetoFront
from src.fitness import (
    image_matrix_average,
    image_matrix_negative_boundary_average,
    image_matrix_sum,
)
from src.pareto import Solution
from src.sliding_window import (
    Increment,
    Shape,
    SlidingWindowProcessor,
    SlidingWindowScanner,
    Stride,
)
from src.utils import (
    get_gradio_output_images,
    get_predict_mask,
    get_resized_image,
    run_gradio_server,
)

RESIZED_DIM = (256, 256)
WINDOW_WIDTH = 20  # Minimum Width of the window to slide over RESIZED_DIM
INCREMENT_FACTOR = 5  # Factor to set the increment of the window


def get_sliding_window_processor(predict_mask, width, height):
    """
    This function is used to create a SlidingWindowProcessor
    """
    shape = Shape(width=width, height=height)
    stride = Stride(
        horizontal=max(int(width / 2), 1),
        vertical=max(int(height / 2), 1),
    )
    increment = Increment(
        width=max(int(width / INCREMENT_FACTOR), 1),
        height=max(int(height / INCREMENT_FACTOR), 1),
    )
    scanner = SlidingWindowScanner(predict_mask, shape, stride)
    return SlidingWindowProcessor(scanner, increment)


def process_image(
    image, prompts, crop_width_ratio, crop_height_ratio, foreground_object_ratio
):
    """
    Main function that called by the gradio.
    """
    # resize
    image_resized, coordinate_transformer = get_resized_image(image, RESIZED_DIM)

    # predict_mask from API
    predict_mask = get_predict_mask(image_resized, prompts, foreground_object_ratio)

    resized_width_ratio, resized_height_ratio = (
        coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, WINDOW_WIDTH
        )
    )
    processor = get_sliding_window_processor(
        predict_mask, resized_width_ratio, resized_height_ratio
    )
    pareto_front = DividedParetoFront(solution_dimensions=3, num_subsets=10)

    for window in processor.generate_windows():
        solution_data = np.array(
            [
                image_matrix_sum(window.sub_image_matrix),
                image_matrix_average(window.sub_image_matrix),
                image_matrix_negative_boundary_average(window.sub_image_matrix),
            ]
        )
        solution = Solution(solution_data, window)
        pareto_front.add_solution(solution)

    return get_gradio_output_images(
        image,
        predict_mask,
        pareto_front,
        coordinate_transformer,
        crop_width_ratio,
        crop_height_ratio,
    )


def get_pareto_front(
    image, prompts, crop_width_ratio, crop_height_ratio, foreground_object_ratio
):
    """
    Get the pareto front for the given image.
    """
    # resize
    image_resized, coordinate_transformer = get_resized_image(image, RESIZED_DIM)

    # predict_mask from API
    predict_mask = get_predict_mask(image_resized, prompts, foreground_object_ratio)

    resized_width_ratio, resized_height_ratio = (
        coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, WINDOW_WIDTH
        )
    )
    processor = get_sliding_window_processor(
        predict_mask, resized_width_ratio, resized_height_ratio
    )
    pareto_front = DividedParetoFront(solution_dimensions=3, num_subsets=10)

    for window in processor.generate_windows():
        solution_data = np.array(
            [
                image_matrix_sum(window.sub_image_matrix),
                image_matrix_average(window.sub_image_matrix),
                image_matrix_negative_boundary_average(window.sub_image_matrix),
            ]
        )
        solution = Solution(solution_data, window)
        pareto_front.add_solution(solution)

    return pareto_front


if __name__ == "__main__":
    run_gradio_server(process_image)
