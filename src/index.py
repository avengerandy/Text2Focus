import cv2
import numpy as np
import requests
import json
from typing import Tuple
from sliding_window import SlidingWindowProcessor, Shape, Stride, Increment, Window
from pareto import Solution, ParetoFront
from fitness import total_sum, total_positive_ratio


API_URL = "http://pyramid:8081/predict"
IMAGE_PATH = "./src/example.jpg"


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

def get_crop(pred_mask: np.ndarray) -> Window:

    pred_mask = np.array(pred_mask, dtype=np.float32) * 255

    shape = Shape(height=100, width=100)
    stride = Stride(vertical=50, horizontal=50)
    increment = Increment(20, 20)
    processor = SlidingWindowProcessor(pred_mask, shape, stride, increment)

    pareto_front = ParetoFront(solution_dimensions=2)
    for window in processor.process():
        total_sum_result = total_sum(window.sub_array)
        total_positive_ratio_result = total_positive_ratio(window.sub_array, 1)

        solution_data = np.array([total_sum_result, total_positive_ratio_result])
        solution = Solution(solution_data, window)

        pareto_front.add_solution(solution)

    best_solution = pareto_front.get_elbow_point()
    print(f"best solution: {best_solution}")
    print(f"pareto solutions count {len(pareto_front.get_pareto_solutions())}")

    return best_solution.get_metadata()


if __name__ == '__main__':
    try:
        image_rgb = load_image(IMAGE_PATH)
        result = send_image_to_api(image_rgb)
        pred_mask = result.get('pred_mask')

        if pred_mask is None:
            raise Exception("No 'pred_mask' in response.")

        best_metadata = get_crop(pred_mask)
        image = image_rgb[
            best_metadata.i:best_metadata.i + best_metadata.window_height,
            best_metadata.j:best_metadata.j + best_metadata.window_width
        ]
        output_path = './src/output_image.jpg'
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)

        print(f"save to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
