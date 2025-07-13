import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

from src.model import SODModel

IMG_SIZE = 256
USE_GPU = True
MODEL_PATH = "./best-model_epoch-204_mae-0.0505_loss-0.1370.pth"

device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SODModel().to(device)
chkpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(chkpt["model"])
model.eval()


def preprocess_image(input_array: np.ndarray) -> torch.Tensor:
    input_array = input_array.astype(np.float32)

    min_val = np.min(input_array)
    max_val = np.max(input_array)
    normalized_array = (input_array - min_val) / (max_val - min_val)

    resized_array = cv2.resize(normalized_array, (IMG_SIZE, IMG_SIZE))
    input_tensor = torch.tensor(resized_array).permute(2, 0, 1).unsqueeze(0).to(device)

    return input_tensor


def get_image_pred_mask(input_array: np.ndarray) -> list:
    original_size = input_array.shape[:2]
    input_tensor = preprocess_image(input_array)

    with torch.no_grad():
        pred_masks, _ = model(input_tensor)

    pred_mask = pred_masks.squeeze().cpu().numpy()
    pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))

    return pred_mask_resized.tolist()


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_array = np.array(data["image"], dtype=np.float32)

        pred_mask = get_image_pred_mask(image_array)

        return jsonify({"pred_mask": pred_mask})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
