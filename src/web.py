import gradio as gr
import cv2
import numpy as np
from src.utils import convert_to_json, post_json_to_api
from src.sliding_window import SlidingWindowProcessor, Shape, Stride, Increment
from src.pareto import Solution
from src.fitness import total_sum, total_positive_ratio, total_cut_ratio
from src.accelerator import CoordinateTransformer, DividedParetoFront

# Constants
API_URL_PYRAMID = "http://pyramid:8081/predict"
PROMPTS_DEFAULT = "anime face, human face, cartoon face, human head"
RESIZED_IMAGE_DIM = (256, 256)  # (Width, Height)
WINDOW_WIDTH = 20  # Width of the window to slide over RESIZED_IMAGE_DIM
INCREMENT_FACTOR = 5  # Factor to set the increment of the window

def get_focus_metadata(pred_mask: np.ndarray, crop_ratio: tuple, coordinate_transformer: CoordinateTransformer, fitness_weight: list) -> dict:
    """Process the predicted mask to get the best crop window."""
    width, height = coordinate_transformer.convert_original_ratio_to_resized(crop_ratio[0], crop_ratio[1], WINDOW_WIDTH)
    shape = Shape(width=width, height=height)
    stride = Stride(horizontal=max(int(width / 2), 1), vertical=max(int(height / 2), 1))
    increment = Increment(width=max(int(width / INCREMENT_FACTOR), 1), height=max(int(height / INCREMENT_FACTOR), 1))
    processor = SlidingWindowProcessor(pred_mask, shape, stride, increment)
    pareto_front = DividedParetoFront(solution_dimensions=3, num_subsets=10)

    for window in processor.process():
        total_sum_result = total_sum(window.sub_array)
        total_positive_ratio_result = total_positive_ratio(window.sub_array)
        total_cut_ratio_result = total_cut_ratio(window.sub_array)

        solution_data = np.array([
            total_sum_result,
            total_positive_ratio_result,
            total_cut_ratio_result
        ])
        solution = Solution(solution_data, window)
        pareto_front.add_solution(solution)

    best_solution = pareto_front.get_point_by_weight(fitness_weight)

    return best_solution.get_metadata()

def process_image(image, prompts, crop_width_ratio, crop_height_ratio, use_owlv2, fitness_weight_1, fitness_weight_2, fitness_weight_3):
    try:
        # Resize the image
        height, width, _ = image.shape
        coordinate_transformer = CoordinateTransformer(width, height, *RESIZED_IMAGE_DIM)
        image_resized = cv2.resize(image, RESIZED_IMAGE_DIM)

        # Send image to API and get pyramid mask
        result = post_json_to_api(API_URL_PYRAMID, {'image': convert_to_json(image_resized)})
        pred_mask = result.get('pred_mask')
        if pred_mask is None:
            raise Exception("No 'pred_mask' in response.")
        pred_mask = np.array(pred_mask, dtype=np.float32)

        # Optionally send image to API_URL_OWLV2
        if use_owlv2:
            result = post_json_to_api("http://owlv2:8082/predict", {'image': convert_to_json(image_resized), 'prompts': prompts})
            pred_mask_2 = result.get('pred_mask')
            if pred_mask_2 is None:
                raise Exception("No 'pred_mask_2' in response.")
            pred_mask = pred_mask + np.array(pred_mask_2, dtype=np.float32)

        # Get the best metadata based on the pareto front
        best_metadata = get_focus_metadata(
            pred_mask,
            (crop_width_ratio, crop_height_ratio),
            coordinate_transformer,
            [fitness_weight_1, fitness_weight_2, fitness_weight_3]
        )

        # Extract and crop the original image based on the best metadata
        i, j = coordinate_transformer.convert_resized_to_original(best_metadata.i, best_metadata.j)
        crop_width, crop_height = coordinate_transformer.convert_resized_to_original(best_metadata.window_width, best_metadata.window_height)
        crop_height = int(crop_width * (crop_height_ratio / crop_width_ratio))
        cropped_image = image[j:j + crop_height, i:i + crop_width]

        return cropped_image

    except Exception as e:
        return f"An error occurred while processing the image: {e}"

def gradio_interface():
    # Create a Gradio interface
    with gr.Blocks() as demo:
        with gr.Row():
            prompts_input = gr.Textbox(value=PROMPTS_DEFAULT, label="Prompts")

            # Crop ratio inputs for width and height as text boxes
            crop_width_ratio_input = gr.Number(value=1.0, label="Crop Width Ratio", interactive=True)
            crop_height_ratio_input = gr.Number(value=1.0, label="Crop Height Ratio", interactive=True)

            fitness_weight_1 = gr.Number(value=1.0, label="total_sum weight", interactive=True)
            fitness_weight_2 = gr.Number(value=1.0, label="total_positive_ratio weight", interactive=True)
            fitness_weight_3 = gr.Number(value=1.0, label="total_cut_ratio weight", interactive=True)

            use_owlv2_checkbox = gr.Checkbox(value=True, label="Use OWLV2 API")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            output_image = gr.Image(type="numpy", label="Cropped Image")

        # Submit button
        submit_btn = gr.Button("Process Image")
        submit_btn.click(
            process_image,
            inputs=[
                image_input,
                prompts_input,
                crop_width_ratio_input,
                crop_height_ratio_input,
                use_owlv2_checkbox,
                fitness_weight_1,
                fitness_weight_2,
                fitness_weight_3
            ],
            outputs=[output_image]
        )

    # Launch Gradio on 0.0.0.0 to allow external access
    demo.launch(server_name="0.0.0.0", server_port=8080)

if __name__ == "__main__":
    gradio_interface()
