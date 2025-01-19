import unittest
import subprocess
import time
import requests
import os
import cv2
import numpy as np

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'INTEL'

IMAGE_PATH = "./example.jpg"
OUTPUT_PATH = "./example_mask.jpg"
EXPECTED_MASK_PATH = "./expected_mask.jpg"


class TestImagePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server_process = subprocess.Popen(
            ['python', 'server.py'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        cls.server_process.daemon = True
        time.sleep(5)

    def test_image_prediction(self):
        client_process = subprocess.Popen(
            ['python', 'client.py'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        time.sleep(2)

        stdout, stderr = client_process.communicate()

        self.assertTrue(os.path.exists(OUTPUT_PATH), "Predicted mask image not saved.")
        self.compare_masks(EXPECTED_MASK_PATH, OUTPUT_PATH)

    def compare_masks(self, expected_mask_path, predicted_mask_path):
        expected_mask = cv2.imread(expected_mask_path, cv2.IMREAD_GRAYSCALE)
        predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

        self.assertIsNotNone(expected_mask, f"Expected mask not found at {expected_mask_path}")
        self.assertIsNotNone(predicted_mask, f"Predicted mask not found at {predicted_mask_path}")

        mse_value = np.mean((expected_mask - predicted_mask) ** 2)
        self.assertLess(mse_value, 10, "MSE is too high, the masks are too different.")

    @classmethod
    def tearDown(self):
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)


if __name__ == '__main__':
    unittest.main()
