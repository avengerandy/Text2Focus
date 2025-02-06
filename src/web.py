import gradio as gr
import cv2
import numpy as np
from src.utils import convert_to_json, post_json_to_api
from src.sliding_window import SlidingWindowProcessor, Shape, Stride, Increment
from src.pareto import Solution, IParetoFront
from src.fitness import total_sum, total_positive_ratio, total_cut_ratio
from src.accelerator import CoordinateTransformer, DividedParetoFront

# Constants
API_URL_PYRAMID = "http://pyramid:8081/predict"
PROMPTS_DEFAULT = "anime face, human face, cartoon face, human head"
RESIZED_IMAGE_DIM = (256, 256)  # (Width, Height)
WINDOW_WIDTH = 20  # Width of the window to slide over RESIZED_IMAGE_DIM
INCREMENT_FACTOR = 5  # Factor to set the increment of the window


def get_pareto_front(
    pred_mask: np.ndarray,
    crop_ratio: tuple,
    coordinate_transformer: CoordinateTransformer,
) -> IParetoFront:
    """Process the predicted mask to get the best crop window."""
    width, height = coordinate_transformer.convert_original_ratio_to_resized(
        crop_ratio[0], crop_ratio[1], WINDOW_WIDTH
    )
    shape = Shape(width=width, height=height)
    stride = Stride(horizontal=max(int(width / 2), 1), vertical=max(int(height / 2), 1))
    increment = Increment(
        width=max(int(width / INCREMENT_FACTOR), 1),
        height=max(int(height / INCREMENT_FACTOR), 1),
    )
    processor = SlidingWindowProcessor(pred_mask, shape, stride, increment)
    pareto_front = DividedParetoFront(solution_dimensions=3, num_subsets=3)

    for window in processor.process():
        total_sum_result = total_sum(window.sub_image_matrix)
        total_positive_ratio_result = total_positive_ratio(window.sub_image_matrix)
        total_cut_ratio_result = total_cut_ratio(window.sub_image_matrix)

        solution_data = np.array(
            [total_sum_result, total_positive_ratio_result, total_cut_ratio_result]
        )
        solution = Solution(solution_data, window)
        pareto_front.add_solution(solution)

    return pareto_front


def process_image(image, prompts, crop_width_ratio, crop_height_ratio, use_owlv2):
    try:
        # Resize the image
        height, width, _ = image.shape
        coordinate_transformer = CoordinateTransformer(
            width, height, *RESIZED_IMAGE_DIM
        )
        image_resized = cv2.resize(image, RESIZED_IMAGE_DIM)

        # Send image to API and get pyramid mask
        result = post_json_to_api(
            API_URL_PYRAMID, {"image": convert_to_json(image_resized)}
        )
        pred_mask = result.get("pred_mask")
        if pred_mask is None:
            raise Exception("No 'pred_mask' in response.")
        pred_mask = np.array(pred_mask, dtype=np.float32)

        # Optionally send image to API_URL_OWLV2
        if use_owlv2:
            result = post_json_to_api(
                "http://owlv2:8082/predict",
                {"image": convert_to_json(image_resized), "prompts": prompts},
            )
            pred_mask_2 = result.get("pred_mask")
            if pred_mask_2 is None:
                raise Exception("No 'pred_mask_2' in response.")
            pred_mask = pred_mask + np.array(pred_mask_2, dtype=np.float32)

        # Get the best metadata based on the pareto front.
        pareto_front = get_pareto_front(
            pred_mask, (crop_width_ratio, crop_height_ratio), coordinate_transformer
        )

        solutions = pareto_front.select_representative_solutions(3)
        solutions.append(pareto_front.get_point_by_weight([1, 1, 1]))

        # Extract and crop the original image based on the metadata
        cropped_images = []
        for solution in solutions:
            metadata = solution.get_metadata()
            i, j = coordinate_transformer.convert_resized_to_original(
                metadata.i, metadata.j
            )
            crop_width, crop_height = (
                coordinate_transformer.convert_resized_to_original(
                    metadata.window_width, metadata.window_height
                )
            )
            crop_height = int(crop_width * (crop_height_ratio / crop_width_ratio))
            cropped_images.append(image[j : j + crop_height, i : i + crop_width])

        cropped_images.append(
            cv2.resize(pred_mask / 3, (image.shape[1], image.shape[0]))
        )

        return cropped_images

    except Exception as e:
        return f"An error occurred while processing the image: {e}"


def gradio_interface():
    # Create a Gradio interface
    with gr.Blocks() as demo:
        with gr.Row():
            prompts_input = gr.Textbox(value=PROMPTS_DEFAULT, label="Prompts")

            # Crop ratio inputs for width and height as text boxes
            crop_width_ratio_input = gr.Number(
                value=1.0, label="Crop Width Ratio", interactive=True
            )
            crop_height_ratio_input = gr.Number(
                value=1.0, label="Crop Height Ratio", interactive=True
            )

            use_owlv2_checkbox = gr.Checkbox(value=True, label="Use OWLV2 API")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            mask_image = gr.Image(type="numpy", label="mask Image")

        with gr.Row():
            output_image1 = gr.Image(
                type="numpy", label="Cropped Image (representative solution)"
            )
            output_image2 = gr.Image(
                type="numpy", label="Cropped Image (representative solution)"
            )
            output_image3 = gr.Image(
                type="numpy", label="Cropped Image (representative solution)"
            )
            output_image4 = gr.Image(
                type="numpy", label="Cropped Image (get by middle point weight)"
            )

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
            ],
            outputs=[
                output_image1,
                output_image2,
                output_image3,
                output_image4,
                mask_image,
            ],
        )

    # Launch Gradio on 0.0.0.0 to allow external access
    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    gradio_interface()
