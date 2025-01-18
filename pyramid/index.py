import cv2
import torch
import numpy as np
from src.model import SODModel

IMG_PATH = './example.jpg'
MODEL_PATH = './best-model_epoch-204_mae-0.0505_loss-0.1370.pth'
IMG_SIZE = 256
USE_GPU = True

def run_inference_on_single_image():
    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SODModel().to(device)
    chkpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.eval()

    img = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img_rgb.shape
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_masks, _ = model(img_tensor)

    pred_mask = pred_masks.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)

    output_path = './predicted_mask.jpg'
    cv2.imwrite(output_path, pred_mask)
    print(f"Predicted mask saved to {output_path}")

if __name__ == '__main__':
    run_inference_on_single_image()
