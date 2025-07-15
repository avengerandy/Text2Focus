"""
This module is used to compare the performance of gene algorithm and sliding window method
"""

# pylint: disable=R0914
# R0914: Too many local variables

import csv
import os
import time

from src.example_server_gene_window import get_pareto_front as gene_get_pareto_front
from src.example_server_sliding_processor import (
    get_pareto_front as sliding_get_pareto_front,
)
from src.pareto import Solution
from src.utils import load_image

if not os.path.exists("src") or not os.path.exists("experiment"):
    raise RuntimeError("Please run this script from the project root directory.")


def process_images_in_folder(
    data_folder,
    prompts,
    output_csv,
    crop_width_ratio=1,
    crop_height_ratio=1,
    foreground_object_ratio=0.5,
):
    """
    Processes all images in the specified folder, combines the maximum values
    from three dimensions into a new solution, calculates its PR values using
    the sliding window method, and compares the execution time of both methods.

    Parameters:
        data_folder (str): Path to the folder containing images.
        prompts (str): Prompts for the prediction mask.
        crop_width_ratio (float): Crop width ratio.
        crop_height_ratio (float): Crop height ratio.
        foreground_object_ratio (float): Foreground object ratio.
        output_csv (str): Path to the output CSV file.
    """
    results = []

    for filename in os.listdir(data_folder):
        image_path = os.path.join(data_folder, filename)
        image = load_image(image_path)

        # Measure time for genetic algorithm
        start_time = time.time()
        pareto_front_gene = gene_get_pareto_front(
            image, prompts, crop_width_ratio, crop_height_ratio, foreground_object_ratio
        )
        gene_time = time.time() - start_time

        # Measure time for sliding window method
        start_time = time.time()
        pareto_front_sliding = sliding_get_pareto_front(
            image, prompts, crop_width_ratio, crop_height_ratio, foreground_object_ratio
        )
        sliding_time = time.time() - start_time

        # Get PR values for each dimension
        max_dim1_solution = pareto_front_gene.get_point_by_weight([1, 0, 0])
        max_dim2_solution = pareto_front_gene.get_point_by_weight([0, 1, 0])
        max_dim3_solution = pareto_front_gene.get_point_by_weight([0, 0, 1])
        combined_solution = Solution(
            [
                max_dim1_solution.get_values()[0],
                max_dim2_solution.get_values()[1],
                max_dim3_solution.get_values()[2],
            ],
            metadata=None,
        )
        pr_values = pareto_front_sliding.get_percentile_rank(combined_solution)

        print(f"Processed {filename}:")
        print(f"  Genetic Algorithm Time: {gene_time:.2f} seconds")
        print(f"  Sliding Window Time: {sliding_time:.2f} seconds")
        print(f"  PR Values in Sliding Window Method: {pr_values[0]:.2f}")
        print(f"  PR Values in Sliding Window Method: {pr_values[1]:.2f}")
        print(f"  PR Values in Sliding Window Method: {pr_values[2]:.2f}")

        results.append(
            {
                "filename": filename,
                "gene_time": gene_time,
                "sliding_time": sliding_time,
                "pr_dim1": pr_values[0],
                "pr_dim2": pr_values[1],
                "pr_dim3": pr_values[2],
            }
        )

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "filename",
            "gene_time",
            "sliding_time",
            "pr_dim1",
            "pr_dim2",
            "pr_dim3",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    process_images_in_folder(
        "./experiment/dogs",
        "dog",
        "./experiment/experiment_dogs_results.csv",
    )
    print("Processing complete.")
