import cv2
import numpy as np
import requests
import json
from typing import Tuple
from sliding_window import SlidingWindowProcessor, Shape, Stride, Increment, Window
from pareto import Solution, ParetoFront
from fitness import total_sum, total_positive_ratio, total_cut_ratio
from accelerator import CoordinateTransformer

API_URL = "http://pyramid:8081/predict"
IMAGE_PATH = "./src/example.jpg"
OUTPUT_PATH = "./src/example_output.jpg"

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def image_to_json(image_rgb):
    image_rgb = image_rgb.astype(np.float32)
    return image_rgb.tolist()


def send_image_to_api(image_rgb):
    json_data = {
        'image': image_to_json(image_rgb)
    }

    response = requests.post(API_URL, json=json_data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error from API: {response.status_code} - {response.text}")

def get_crop(pred_mask: np.ndarray, coordinateTransformer: CoordinateTransformer) -> Window:

    pred_mask = np.array(pred_mask, dtype=np.float32)

    width, height = coordinateTransformer.convert_original_ratio_to_resized(1, 1, 20)

    shape = Shape(width=width, height=height)
    stride = Stride(horizontal=max(int(width/2), 1), vertical=max(int(height/2), 1))
    increment = Increment(width=max(int(width/5), 1), height=max(int(height/5), 1))
    processor = SlidingWindowProcessor(pred_mask, shape, stride, increment)

    pareto_front = ParetoFront(solution_dimensions=3)
    positive_ratio_threshold = np.max(pred_mask) * 0.01
    for window in processor.process():
        total_sum_result = total_sum(window.sub_array)
        total_positive_ratio_result = total_positive_ratio(window.sub_array, positive_ratio_threshold)
        total_cut_ratio_result = total_cut_ratio(window.sub_array)

        solution_data = np.array([
            total_sum_result,
            total_positive_ratio_result,
            total_cut_ratio_result
        ])
        solution = Solution(solution_data, window)

        pareto_front.add_solution(solution)

    best_solution = pareto_front.get_point_by_weight([1, 1, 1])
    print(f"best solution: {best_solution}")
    print(f"pareto solutions count {len(pareto_front.get_pareto_solutions())}")

    return best_solution.get_metadata()


if __name__ == '__main__':
    try:
        image = load_image(IMAGE_PATH)
        height, width, _ = image.shape
        coordinateTransformer = CoordinateTransformer(width, height, 256, 256)
        image_resized = cv2.resize(image, (256, 256))
        result = send_image_to_api(image_resized)
        pred_mask = result.get('pred_mask')

        if pred_mask is None:
            raise Exception("No 'pred_mask' in response.")

        best_metadata = get_crop(pred_mask, coordinateTransformer)

        i, j = coordinateTransformer.convert_resized_to_original(best_metadata.i, best_metadata.j)
        width, height = coordinateTransformer.convert_resized_to_original(best_metadata.window_width, best_metadata.window_height)
        image = image[j:j + height, i:i + width]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_PATH, image)
        print(f"save to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred: {e}")
