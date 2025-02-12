"""
Utility functions for the example file.
"""

# pylint: disable=E1101
# E1101: Module has no member (cannot correctly identify members of the 'cv2' module)

from typing import Callable

import cv2
import gradio as gr
import numpy as np
import requests

from src.accelerator import CoordinateTransformer
from src.pareto import IParetoFront

API_URL_PYRAMID = "http://pyramid:8081/predict"
API_URL_OWLV2 = "http://owlv2:8082/predict"
PROMPTS_DEFAULT = "anime face, human face, cartoon face, human head"


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path and return it as a numpy array.
    The image is converted to RGB format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image_path: str, image_rgb: np.ndarray) -> None:
    """
    Save an image as a file. The image_rgb is RGB format.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)


def convert_to_json(image_rgb: np.ndarray) -> list:
    """
    Convert an image to a list of floats.
    """
    return image_rgb.astype(np.float32).tolist()


def post_json_to_api(api_url: str, json_data: dict) -> dict:
    """
    Post a JSON data to an API and return the response as a dictionary.
    """
    try:
        response = requests.post(api_url, json=json_data, timeout=10)
        response.raise_for_status()  # Will raise an exception for 4xx/5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error from API: {e}")


def get_resized_image(
    image: np.ndarray, resized_dim: tuple
) -> tuple[np.ndarray, CoordinateTransformer]:
    """
    Resize an image to the specified dimensions.
    Return the resized image and a CoordinateTransformer object to convert coordinates.
    """
    height, width, _ = image.shape
    coordinate_transformer = CoordinateTransformer(width, height, *resized_dim)
    image_resized = cv2.resize(image, resized_dim)

    return image_resized, coordinate_transformer


def get_predict_mask(
    image: np.ndarray,
    prompts: str = PROMPTS_DEFAULT,
    foreground_object_ratio: float = 0.5,
) -> np.ndarray:
    """
    Get a mask image from the pyramid and owl-v2 APIs.
    The mask image is a combination of the two masks with the foreground_object_ratio.
    """
    result = post_json_to_api(API_URL_PYRAMID, {"image": convert_to_json(image)})
    predict_foreground_mask = result.get("pred_mask")
    if predict_foreground_mask is None:
        raise TypeError("pyramid did not return 'pred_mask'.")
    predict_foreground_mask = np.array(predict_foreground_mask, dtype=np.float32)

    result = post_json_to_api(
        API_URL_OWLV2, {"image": convert_to_json(image), "prompts": prompts}
    )
    predict_object_mask = result.get("pred_mask")
    if predict_object_mask is None:
        raise TypeError("owlv2 API did not return 'pred_mask'.")
    predict_object_mask = np.array(predict_object_mask, dtype=np.float32)

    return (
        foreground_object_ratio * predict_foreground_mask
        + (1 - foreground_object_ratio) * predict_object_mask
    )


def get_gradio_output_images(
    image: np.ndarray,
    predict_mask: np.ndarray,
    pareto_front: IParetoFront,
    coordinate_transformer: CoordinateTransformer = None,
    crop_width_ratio: int = 1,
    crop_height_ratio: int = 1,
) -> list:
    """
    Get the cropped images from the original image and the predict mask. The cropped images
    are the representative solutions and the decomposition by weight from the pareto front.
    """
    cropped_images = []
    solutions = pareto_front.select_representative_solutions(3)
    solutions.append(pareto_front.get_point_by_weight([1, 1, 1]))
    for solution in solutions:
        metadata = solution.get_metadata()

        if coordinate_transformer:
            i, j = coordinate_transformer.convert_resized_to_original(
                metadata.i, metadata.j
            )
            crop_width, crop_height = (
                coordinate_transformer.convert_resized_to_original(
                    metadata.window_width, metadata.window_height
                )
            )
            crop_height = int(crop_width * (crop_height_ratio / crop_width_ratio))
        else:
            i, j, crop_width, crop_height = (
                metadata.i,
                metadata.j,
                metadata.window_width,
                metadata.window_height,
            )

        cropped_images.append(image[j : j + crop_height, i : i + crop_width])

    cropped_images.append(cv2.resize(predict_mask, (image.shape[1], image.shape[0])))

    return cropped_images


def run_gradio_server(process_image: Callable) -> None:
    """
    Create Gradio UI for the process_image function.
    """
    with gr.Blocks() as demo:
        with gr.Row():
            prompts_input = gr.Textbox(value=PROMPTS_DEFAULT, label="Prompts")
            crop_width_ratio_input = gr.Number(
                value=1.0, label="Crop Width Ratio", interactive=True
            )
            crop_height_ratio_input = gr.Number(
                value=1.0, label="Crop Height Ratio", interactive=True
            )
            foreground_object_ratio = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.5,
                label="Foreground Object Ratio",
            )

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            mask_image = gr.Image(type="numpy", label="mask Image")

        with gr.Row():
            output_representative_1 = gr.Image(
                type="numpy",
                label="Cropped Image (representative solution)",
            )
            output_representative_2 = gr.Image(
                type="numpy",
                label="Cropped Image (representative solution)",
            )
            output_representative_3 = gr.Image(
                type="numpy",
                label="Cropped Image (representative solution)",
            )
            output_weighted = gr.Image(
                type="numpy",
                label="Cropped Image (decomposition by weight)",
            )

        submit_btn = gr.Button("Process Image")
        submit_btn.click(
            process_image,
            inputs=[
                image_input,
                prompts_input,
                crop_width_ratio_input,
                crop_height_ratio_input,
                foreground_object_ratio,
            ],
            outputs=[
                output_representative_1,
                output_representative_2,
                output_representative_3,
                output_weighted,
                mask_image,
            ],
        )

    demo.launch(server_name="0.0.0.0", server_port=8080)
