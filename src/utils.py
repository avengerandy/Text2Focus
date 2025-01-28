import cv2
import numpy as np
import requests

def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(image_path: str, image_rgb: np.ndarray) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)

def convert_to_json(image_rgb: np.ndarray) -> list:
    return image_rgb.astype(np.float32).tolist()

def post_json_to_api(api_url: str, json_data: dict) -> dict:
    response = requests.post(api_url, json=json_data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error from API: {response.status_code} - {response.text}")
