"""
This module is a example of how to use

* CoordinateTransformer (resize image)
* SlidingWindowScanner (brute force)

to find the best crop for a given image.
"""

# pylint: disable=R0801
# R0801: duplicate-code
# pylint: disable=E1101
# E1101: Module has no member (cannot correctly identify members of the 'cv2' module)

import cv2

from src.fitness import image_matrix_sum
from src.sliding_window import Shape, SlidingWindowScanner, Stride
from src.utils import get_predict_mask, get_resized_image, run_gradio_server

RESIZED_DIM = (256, 256)
STRIDE_DIM = (10, 10)  # slide step size


def get_sliding_window_scanner(predict_mask, resized_width_ratio, resized_height_ratio):
    """
    This function is used to create a SlidingWindowScanner
    """

    width = RESIZED_DIM[0]
    height = int(width * resized_height_ratio / resized_width_ratio)
    if height > RESIZED_DIM[1]:
        height = RESIZED_DIM[1]
        width = int(height * resized_width_ratio / resized_height_ratio)

    shape = Shape(width=width, height=height)
    stride = Stride(horizontal=STRIDE_DIM[0], vertical=STRIDE_DIM[1])

    return SlidingWindowScanner(predict_mask, shape, stride)


def crop_image(
    image, window, crop_height_ratio, crop_width_ratio, coordinate_transformer
):
    """
    Crop the image based on the window and the coordinate transformer.
    """
    i, j = coordinate_transformer.convert_resized_to_original(window.i, window.j)
    crop_width, crop_height = coordinate_transformer.convert_resized_to_original(
        window.window_width, window.window_height
    )
    crop_height = int(crop_width * (crop_height_ratio / crop_width_ratio))

    return image[j : j + crop_height, i : i + crop_width]


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

    # 100 can be any large number.
    # It is just to ensure that the resized ratio is not 0
    # or that the proportions do not change too much.
    resized_width_ratio, resized_height_ratio = (
        coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, 100
        )
    )
    scanner = get_sliding_window_scanner(
        predict_mask, resized_width_ratio, resized_height_ratio
    )

    best_window = None
    maximum = 0
    for window in scanner.generate_windows():
        score = image_matrix_sum(window.sub_image_matrix)
        if best_window is None or score > maximum:
            best_window = window
            maximum = score

    return [
        None,
        None,
        None,
        crop_image(
            image,
            best_window,
            crop_width_ratio,
            crop_height_ratio,
            coordinate_transformer,
        ),
        cv2.resize(predict_mask, (image.shape[1], image.shape[0])),
    ]


if __name__ == "__main__":
    run_gradio_server(process_image)
