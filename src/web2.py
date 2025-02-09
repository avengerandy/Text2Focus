import gradio as gr
import cv2
import numpy as np
import random
from src.utils import convert_to_json, post_json_to_api
from src.sliding_window import Window
from src.pareto import Solution, IParetoFront, ParetoFront
from src.fitness import image_matrix_sum, image_matrix_average, image_matrix_negative_boundary_average
from src.accelerator import CoordinateTransformer, DividedParetoFront

# Constants
API_URL_PYRAMID = "http://pyramid:8081/predict"
PROMPTS_DEFAULT = "anime face, human face, cartoon face, human head"
RESIZED_IMAGE_DIM = (256, 256)  # (Width, Height)
WINDOW_WIDTH = 20  # Width of the window to slide over RESIZED_IMAGE_DIM
INCREMENT_FACTOR = 5  # Factor to set the increment of the window

def generate_random_rectangle(width_ratio, height_ratio):
    width, height = RESIZED_IMAGE_DIM

    rect_width = random.uniform(1, width)
    rect_height = (rect_width * height_ratio) / width_ratio

    if rect_height > height:
        rect_height = height
        rect_width = (rect_height * width_ratio) / height_ratio

    x = random.uniform(0, width - rect_width)
    y = random.uniform(0, height - rect_height)

    return int(x), int(y), int(rect_width), int(rect_height)

def create_random_window(pred_mask, width_ratio, height_ratio):
    i, j, width, height = generate_random_rectangle(
        width_ratio=width_ratio, height_ratio=height_ratio
    )
    sub_image_matrix = pred_mask[j : j + height, i : i + width]
    return Window(
        sub_image_matrix=sub_image_matrix,
        i=i,
        j=j,
        window_width=width,
        window_height=height,
    )

def get_pareto_front(
    pred_mask: np.ndarray,
    crop_ratio: tuple,
    coordinate_transformer: CoordinateTransformer,
) -> IParetoFront:
    """Process the predicted mask to get the best crop window."""
    width, height = coordinate_transformer.convert_original_ratio_to_resized(
        crop_ratio[0], crop_ratio[1], WINDOW_WIDTH
    )
    pareto_front = ParetoFront(solution_dimensions=3)

    fft = 1000
    for temp1 in range(1000):
        pareto_solutions = pareto_front.get_pareto_solutions()
        if fft <= 0:
            break

        if len(pareto_solutions) < 5:
            fft -= 5
            for temp2 in range(5):
                window = create_random_window(pred_mask, width, height)
                solution_data = np.array(
                    [
                        image_matrix_sum(window.sub_image_matrix),
                        image_matrix_average(window.sub_image_matrix),
                        image_matrix_negative_boundary_average(window.sub_image_matrix),
                    ]
                )
                solution = Solution(solution_data, window)
                pareto_solutions.append(solution)

        # crossover
        random_elements = random.sample(pareto_solutions, 2)
        i = int(
            (random_elements[0].get_metadata().i + random_elements[1].get_metadata().i)
            / 2
        )
        j = int(
            (random_elements[0].get_metadata().j + random_elements[1].get_metadata().j)
            / 2
        )
        window_width = int(
            (
                random_elements[0].get_metadata().window_width
                + random_elements[1].get_metadata().window_width
            )
            / 2
        )
        window_height = int(
            (
                random_elements[0].get_metadata().window_height
                + random_elements[1].get_metadata().window_height
            )
            / 2
        )

        if (i + window_width) < 256 and (j + window_height) < 256:
            fft -= 1
            window = Window(
                sub_image_matrix=pred_mask[j : j + window_height, i : i + window_width],
                i=i,
                j=j,
                window_width=window_width,
                window_height=window_height,
            )

            image_matrix_sum_result = image_matrix_sum(window.sub_image_matrix)
            image_matrix_average_result = image_matrix_average(window.sub_image_matrix)
            image_matrix_negative_boundary_average_result = image_matrix_negative_boundary_average(window.sub_image_matrix)
            solution_data = np.array(
                [image_matrix_sum_result, image_matrix_average_result, image_matrix_negative_boundary_average_result]
            )
            solution = Solution(solution_data, window)
            pareto_front.add_solution(solution)

        # mutation
        random_element = random.choice(pareto_solutions)
        i, j, window_width, window_height = generate_random_rectangle(width, height)
        i = int((random_element.get_metadata().i + i) / 2)
        j = int((random_element.get_metadata().j + j) / 2)
        window_width = int(
            (random_element.get_metadata().window_width + window_width) / 2
        )
        window_height = int(
            (random_element.get_metadata().window_height + window_height) / 2
        )
        if i + window_width < 256 and j + window_height < 256:
            fft -= 1
            window = Window(
                sub_image_matrix=pred_mask[j : j + window_height, i : i + window_width],
                i=i,
                j=j,
                window_width=window_width,
                window_height=window_height,
            )
            image_matrix_sum_result = image_matrix_sum(window.sub_image_matrix)
            image_matrix_average_result = image_matrix_average(window.sub_image_matrix)
            image_matrix_negative_boundary_average_result = image_matrix_negative_boundary_average(window.sub_image_matrix)
            solution_data = np.array(
                [image_matrix_sum_result, image_matrix_average_result, image_matrix_negative_boundary_average_result]
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
                type="numpy", label="Cropped Image (representative solution)"
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
