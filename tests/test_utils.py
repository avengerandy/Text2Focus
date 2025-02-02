import unittest
import numpy as np
from unittest.mock import patch, Mock
from src.utils import load_image, save_image, convert_to_json, post_json_to_api

class TestUtils(unittest.TestCase):

    def test_load_image(self):
        with patch('cv2.imread', return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
            image = load_image('dummy_path')
            self.assertEqual(image.shape, (10, 10, 3))

        with patch('cv2.imread', return_value=None):
            with self.assertRaises(FileNotFoundError):
                load_image('dummy_path')

    def test_save_image(self):
        image_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch('cv2.imwrite', return_value=True) as mock_write:
            save_image('dummy_path', image_rgb)
            mock_write.assert_called_once()

    def test_convert_to_json(self):
        image_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        json_data = convert_to_json(image_rgb)
        self.assertIsInstance(json_data, list)
        self.assertEqual(len(json_data), 10)
        self.assertEqual(len(json_data[0]), 10)

    def test_post_json_to_api(self):
        json_data = {"key": "value"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "success"}

        with patch('requests.post', return_value=mock_response) as mock_post:
            response = post_json_to_api('http://dummy_api', json_data)
            self.assertEqual(response, {"response": "success"})
            mock_post.assert_called_once_with('http://dummy_api', json=json_data)

        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        with patch('requests.post', return_value=mock_response):
            with self.assertRaises(Exception):
                post_json_to_api('http://dummy_api', json_data)


if __name__ == "__main__":
    unittest.main()
