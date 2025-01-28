import requests
import cv2
import numpy as np
import json

API_URL = "http://127.0.0.1:8082/predict"
IMAGE_PATH = "./example.jpg"
OUTPUT_PATH = "./example_mask.jpg"

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
        'image': image_to_json(image_rgb),
        'prompts': "anime face, hunman face, cartoon face, hunman head"
    }

    response = requests.post(API_URL, json=json_data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error from API: {response.status_code} - {response.text}")

def save_predicted_mask(pred_mask, output_path):
    pred_mask = np.array(pred_mask, dtype=np.float32)
    pred_mask = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, pred_mask)
    print(f"Predicted mask saved to {output_path}")

if __name__ == '__main__':
    try:
        image_rgb = load_image(IMAGE_PATH)
        result = send_image_to_api(image_rgb)
        pred_mask = result.get('pred_mask')

        if pred_mask is not None:
            save_predicted_mask(pred_mask, OUTPUT_PATH)
        else:
            print("Error: No 'pred_mask' in response.")

    except Exception as e:
        print(f"An error occurred: {e}")
