import argparse

import numpy as np

from src.template import (
    GeneFocusTemplate,
    NSGA2FocusTemplate,
    ScannerFocusTemplate,
    SlidingFocusTemplate,
)
from src.utils import get_gradio_output_images, run_gradio_server

STRATEGY_MAP = {
    "scanner": ScannerFocusTemplate,
    "sliding": SlidingFocusTemplate,
    "genetic": GeneFocusTemplate,
    "nsga2": NSGA2FocusTemplate,
}

focus = None


def process_image(
    image, prompts, crop_width_ratio, crop_height_ratio, foreground_object_ratio
):
    """
    Main function that called by the gradio.
    """
    crop_result = focus.crop(
        image, prompts, crop_width_ratio, crop_height_ratio, foreground_object_ratio
    )

    return get_gradio_output_images(
        image,
        crop_result.predict_mask,
        crop_result.pareto_front,
        crop_result.coordinate_transformer,
        crop_width_ratio,
        crop_height_ratio,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default="genetic",
        choices=list(STRATEGY_MAP.keys()),
        help="Which focus template strategy to use",
    )
    args = parser.parse_args()
    focus = STRATEGY_MAP[args.strategy]()

    run_gradio_server(process_image)
