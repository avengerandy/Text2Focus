import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from flask import Flask, request, jsonify
import numpy as np
import requests

USE_GPU = True
MODEL_NAME = "google/owlv2-base-patch16-ensemble"

device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

processor = Owlv2Processor.from_pretrained(MODEL_NAME)
model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to(device)
model.eval()

def get_image_pred_mask(input_array: np.ndarray, prompts: str) -> list:
    # Prepare the inputs for the model
    inputs = processor(prompts, images=input_array, return_tensors="pt")
    inputs.to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the outputs (bounding boxes and class logits)
    height, width = input_array.shape[0], input_array.shape[1]
    target_sizes = torch.Tensor([[height, width]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)[0]

    # Iterate over the boxes and fill confidence values
    mask = np.zeros((height, width), dtype=np.float32)
    for box, score in zip(results["boxes"], results["scores"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], score.item())

    # normalize the mask
    if np.max(mask) > 0:
        mask /= np.max(mask)

    return mask.tolist()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_array = np.array(data['image'], dtype=np.float32)
        prompts = data['prompts']

        pred_mask = get_image_pred_mask(image_array, prompts)

        return jsonify({
            'pred_mask': pred_mask
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)
