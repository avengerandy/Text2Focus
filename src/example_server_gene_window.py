"""
This module is a example of how to use

* CoordinateTransformer (resize image)
* GeneWindowGenerator (genetic algorithm)
* ParetoFront (brute force)

to find the best crop for a given image.
"""

# pylint: disable=R0801
# R0801: duplicate-code

import numpy as np

from src.accelerator import GeneWindowGenerator
from src.fitness import (
    image_matrix_average,
    image_matrix_negative_boundary_average,
    image_matrix_sum,
)
from src.pareto import ParetoFront, Solution
from src.utils import (
    get_gradio_output_images,
    get_predict_mask,
    get_resized_image,
    run_gradio_server,
)

RESIZED_DIM = (256, 256)


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

    # use genetic algorithm to find the best window
    pareto_front = ParetoFront(solution_dimensions=3)
    # 100 can be any large number.
    # It is just to ensure that the resized ratio is not 0
    # or that the proportions do not change too much.
    resized_width_ratio, resized_height_ratio = (
        coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, 100
        )
    )
    gene_window_generator = GeneWindowGenerator(
        predict_mask, resized_width_ratio, resized_height_ratio
    )

    for window in gene_window_generator.generate_windows():
        solution_data = np.array(
            [
                image_matrix_sum(window.sub_image_matrix),
                image_matrix_average(window.sub_image_matrix),
                image_matrix_negative_boundary_average(window.sub_image_matrix),
            ]
        )
        solution = Solution(solution_data, window)
        pareto_front.add_solution(solution)
        gene_window_generator.population = [
            solution.get_metadata() for solution in pareto_front.get_pareto_solutions()
        ]

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

    # use genetic algorithm to find the best window
    pareto_front = ParetoFront(solution_dimensions=3)
    # 100 can be any large number.
    # It is just to ensure that the resized ratio is not 0
    # or that the proportions do not change too much.
    resized_width_ratio, resized_height_ratio = (
        coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, 100
        )
    )
    gene_window_generator = GeneWindowGenerator(
        predict_mask, resized_width_ratio, resized_height_ratio
    )

    for window in gene_window_generator.generate_windows():
        solution_data = np.array(
            [
                image_matrix_sum(window.sub_image_matrix),
                image_matrix_average(window.sub_image_matrix),
                image_matrix_negative_boundary_average(window.sub_image_matrix),
            ]
        )
        solution = Solution(solution_data, window)
        pareto_front.add_solution(solution)
        gene_window_generator.population = [
            solution.get_metadata() for solution in pareto_front.get_pareto_solutions()
        ]

    return pareto_front


if __name__ == "__main__":
    run_gradio_server(process_image)
