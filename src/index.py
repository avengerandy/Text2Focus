import cv2
import numpy as np
from src.utils import load_image, save_image, convert_to_json, post_json_to_api
from src.sliding_window import SlidingWindowScanner, SlidingWindowProcessor, Shape, Stride, Increment
from src.pareto import Solution
from src.fitness import image_matrix_sum, image_matrix_average, image_matrix_negative_boundary_average
from src.accelerator import CoordinateTransformer, DividedParetoFront

# Constants
API_URL_PYRAMID = "http://pyramid:8081/predict"
API_URL_OWLV2 = "http://owlv2:8082/predict"
IMAGE_PATH = "./src/example.jpg"
OUTPUT_PATH = "./src/example_output.jpg"
PROMPTS = "anime face, hunman face, cartoon face, hunman head"
RESIZED_IMAGE_DIM = (256, 256)  # (Width, Height)
CROP_RATIO = (1, 1)  # (Width, Height)
WINDOW_WIDTH = 20 # Width of the window to slide over RESIZED_IMAGE_DIM
INCREMENT_FACTOR = 5 # Factor to set the increment of the window

def get_focus_metadata(pred_mask: np.ndarray, crop_ratio: tuple, coordinate_transformer: CoordinateTransformer) -> dict:
    """Process the predicted mask to get the best crop window."""
    width, height = coordinate_transformer.convert_original_ratio_to_resized(crop_ratio[0], crop_ratio[1], WINDOW_WIDTH)
    shape = Shape(width=width, height=height)
    stride = Stride(horizontal=max(int(width / 2), 1), vertical=max(int(height / 2), 1))
    increment = Increment(width=max(int(width / INCREMENT_FACTOR), 1), height=max(int(height / INCREMENT_FACTOR), 1))
    scanner = SlidingWindowScanner(pred_mask, shape, stride)
    processor = SlidingWindowProcessor(scanner, increment)

    pareto_front = DividedParetoFront(solution_dimensions=3, num_subsets=10)

    for window in processor.generate_windows():
        image_matrix_sum_result = image_matrix_sum(window.sub_image_matrix)
        image_matrix_average_result = image_matrix_average(window.sub_image_matrix)
        image_matrix_negative_boundary_average_result = image_matrix_negative_boundary_average(window.sub_image_matrix)

        solution_data = np.array([
            image_matrix_sum_result,
            image_matrix_average_result,
            image_matrix_negative_boundary_average_result
        ])
        solution = Solution(solution_data, window)
        pareto_front.add_solution(solution)

    best_solution = pareto_front.get_point_by_weight([1, 1, 1])
    return best_solution.get_metadata()

def main():
    try:
        # Load and resize the image
        image = load_image(IMAGE_PATH)
        height, width, _ = image.shape
        coordinate_transformer = CoordinateTransformer(width, height, *RESIZED_IMAGE_DIM)
        image_resized = cv2.resize(image, RESIZED_IMAGE_DIM)

        # Send image to API and get pyramid mask
        result = post_json_to_api(API_URL_PYRAMID, {'image': convert_to_json(image_resized)})
        pred_mask = result.get('pred_mask')
        if pred_mask is None:
            raise Exception("No 'pred_mask' in response.")
        pred_mask = np.array(pred_mask, dtype=np.float32)

        # Send image to API and get owlv2 mask
        result = post_json_to_api(API_URL_OWLV2, {'image': convert_to_json(image_resized), 'prompts': PROMPTS})
        pred_mask_v2 = result.get('pred_mask')
        if pred_mask_v2 is None:
            raise Exception("No 'pred_mask_v2' in response.")
        pred_mask = pred_mask + np.array(pred_mask_v2, dtype=np.float32)

        # Get the best focus metadata
        best_metadata = get_focus_metadata(pred_mask, CROP_RATIO, coordinate_transformer)

        # Extract and crop the original image based on the best metadata
        i, j = coordinate_transformer.convert_resized_to_original(best_metadata.i, best_metadata.j)
        crop_width, crop_height = coordinate_transformer.convert_resized_to_original(best_metadata.window_width, best_metadata.window_height)
        crop_height = int(crop_width * (CROP_RATIO[1] / CROP_RATIO[0]))
        cropped_image = image[j:j + crop_height, i:i + crop_width]

        # Save the cropped image
        save_image(OUTPUT_PATH, cropped_image)
        print(f"Saved cropped image to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")

if __name__ == '__main__':
    main()
