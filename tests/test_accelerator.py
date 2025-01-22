import unittest
from src.accelerator import CoordinateTransformer

class TestCoordinateTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = CoordinateTransformer(original_width=1000, original_height=500, resized_width=256, resized_height=256)

    def test_convert_resized_to_original(self):
        resized_x, resized_y = 128, 128
        original_x, original_y = self.transformer.convert_resized_to_original(resized_x, resized_y)
        self.assertEqual(original_x, 500)
        self.assertEqual(original_y, 250)

        original_x, original_y = self.transformer.convert_resized_to_original(resized_x, resized_y, to_int=False)
        self.assertEqual(original_x, 500.0)
        self.assertEqual(original_y, 250.0)

    def test_convert_original_to_resized(self):
        original_x, original_y = 500, 250
        resized_x, resized_y = self.transformer.convert_original_to_resized(original_x, original_y)
        self.assertEqual(resized_x, 128)
        self.assertEqual(resized_y, 128)

        resized_x, resized_y = self.transformer.convert_original_to_resized(original_x, original_y, to_int=False)
        self.assertEqual(resized_x, 128.0)
        self.assertEqual(resized_y, 128.0)

    def test_convert_original_ratio_to_resized(self):
        ratio_x, ratio_y, except_x = 16, 9, 256
        resized_x, resized_y = self.transformer.convert_original_ratio_to_resized(ratio_x, ratio_y, except_x)
        self.assertEqual(resized_x, 256)
        self.assertEqual(resized_y, 288)

        resized_x, resized_y = self.transformer.convert_original_ratio_to_resized(ratio_x, ratio_y, except_x, to_int=False)
        self.assertEqual(resized_x, 256.0)
        self.assertEqual(resized_y, 288.0)

    def test_convert_resized_ratio_to_original(self):
        ratio_x, ratio_y, except_x = 16, 9, 256
        original_x, original_y = self.transformer.convert_resized_ratio_to_original(ratio_x, ratio_y, except_x)
        self.assertEqual(original_x, 256)
        self.assertEqual(original_y, 72)

        original_x, original_y = self.transformer.convert_resized_ratio_to_original(ratio_x, ratio_y, except_x, to_int=False)
        self.assertEqual(original_x, 256.0)
        self.assertEqual(original_y, 72.0)

if __name__ == '__main__':
    unittest.main()
