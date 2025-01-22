class CoordinateTransformer:
    def __init__(self, original_width: int, original_height: int, resized_width: int, resized_height: int):
        self.scale_x = float(original_width) / resized_width
        self.scale_y = float(original_height) / resized_height

    def convert_resized_to_original(self, x: int, y: int, to_int=True) -> tuple:
        original_x = x * self.scale_x
        original_y = y * self.scale_y

        if to_int:
            return int(original_x), int(original_y)
        return original_x, original_y

    def convert_original_to_resized(self, x: int, y: int, to_int=True) -> tuple:
        resized_x = x / self.scale_x
        resized_y = y / self.scale_y

        if to_int:
            return int(resized_x), int(resized_y)
        return resized_x, resized_y

    def convert_original_ratio_to_resized(self, ratio_x: int, ratio_y: int, except_x: int, to_int=True) -> tuple:
        resized_x, resized_y = self.convert_original_to_resized(ratio_x, ratio_y, to_int=False)
        resized_y = resized_y * (except_x / resized_x)

        if to_int:
            return int(except_x), int(resized_y)
        return except_x, resized_y

    def convert_resized_ratio_to_original(self, ratio_x: int, ratio_y: int, except_x: int, to_int=True) -> tuple:
        original_x, original_y = self.convert_resized_to_original(ratio_x, ratio_y, to_int=False)
        original_y = original_y * (except_x / original_x)

        if to_int:
            return int(except_x), int(original_y)
        return except_x, original_y
