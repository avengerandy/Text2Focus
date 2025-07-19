from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.accelerator import (
    CoordinateTransformer,
    DividedParetoFront,
    GeneWindowGenerator,
    NSGA2WindowGenerator,
)
from src.fitness import (
    image_matrix_average,
    image_matrix_negative_boundary_average,
    image_matrix_sum,
)
from src.pareto import IParetoFront, ParetoFront, Solution
from src.sliding_window import (
    Increment,
    Shape,
    SlidingWindowProcessor,
    SlidingWindowScanner,
    Stride,
)
from src.utils import get_predict_mask, get_resized_image

RESIZED_DIM = (256, 256)


@dataclass(frozen=True)
class CropResult:
    pareto_front: IParetoFront
    predict_mask: np.ndarray
    coordinate_transformer: CoordinateTransformer


class Test2FocusTemplate(ABC):
    def __init__(self):
        self._solution_dimensions: int = 3
        self._fitness: array = [
            image_matrix_sum,
            image_matrix_average,
            image_matrix_negative_boundary_average,
        ]
        self._pareto_front: IParetoFront | None = None
        self._window_generator: IWindowGenerator | None = None

    @abstractmethod
    def init_pareto_front(self):
        pass

    @abstractmethod
    def init_window_generator(
        self, predict_mask, resized_width_ratio, resized_height_ratio
    ):
        pass

    @abstractmethod
    def generate_resized_ratio(
        self, coordinate_transformer, crop_width_ratio, crop_height_ratio
    ):
        pass

    def after_window_generate(self):
        pass

    def crop(
        self,
        image,
        prompts,
        crop_width_ratio,
        crop_height_ratio,
        foreground_object_ratio,
    ):
        image_resized, coordinate_transformer = get_resized_image(image, RESIZED_DIM)
        predict_mask = get_predict_mask(image_resized, prompts, foreground_object_ratio)

        resized_width_ratio, resized_height_ratio = self.generate_resized_ratio(
            coordinate_transformer, crop_width_ratio, crop_height_ratio
        )

        self.init_pareto_front()
        self.init_window_generator(
            predict_mask, resized_width_ratio, resized_height_ratio
        )

        for window in self._window_generator.generate_windows():
            solution_data = np.array(
                [
                    self._fitness[0](window.sub_image_matrix),
                    self._fitness[1](window.sub_image_matrix),
                    self._fitness[2](window.sub_image_matrix),
                ]
            )
            solution = Solution(solution_data, window)
            self._pareto_front.add_solution(solution)
            self.after_window_generate()

        return CropResult(
            pareto_front=self._pareto_front,
            predict_mask=predict_mask,
            coordinate_transformer=coordinate_transformer,
        )


class GeneFocusTemplate(Test2FocusTemplate):

    def init_pareto_front(self):
        self._pareto_front = ParetoFront(solution_dimensions=self._solution_dimensions)

    def init_window_generator(
        self, predict_mask, resized_width_ratio, resized_height_ratio
    ):
        self._window_generator = GeneWindowGenerator(
            predict_mask, resized_width_ratio, resized_height_ratio
        )

    def generate_resized_ratio(
        self, coordinate_transformer, crop_width_ratio, crop_height_ratio
    ):
        # 100 can be any large number.
        # It is just to ensure that the resized ratio is not 0
        # or that the proportions do not change too much.
        return coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, 100
        )

    def after_window_generate(self):
        self._window_generator.population = [
            solution.get_metadata()
            for solution in self._pareto_front.get_pareto_solutions()
        ]


class NSGA2FocusTemplate(Test2FocusTemplate):

    def init_pareto_front(self):
        self._pareto_front = ParetoFront(solution_dimensions=self._solution_dimensions)

    def init_window_generator(
        self, predict_mask, resized_width_ratio, resized_height_ratio
    ):
        self._window_generator = NSGA2WindowGenerator(
            predict_mask,
            resized_width_ratio,
            resized_height_ratio,
            fitness_funcs=self._fitness,
        )

    def generate_resized_ratio(
        self, coordinate_transformer, crop_width_ratio, crop_height_ratio
    ):
        # 100 can be any large number.
        # It is just to ensure that the resized ratio is not 0
        # or that the proportions do not change too much.
        return coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, 100
        )


class SlidingFocusTemplate(Test2FocusTemplate):

    WINDOW_WIDTH = 20
    INCREMENT_FACTOR = 5

    def init_pareto_front(self):
        self._pareto_front = DividedParetoFront(
            solution_dimensions=self._solution_dimensions, num_subsets=10
        )

    def generate_resized_ratio(
        self, coordinate_transformer, crop_width_ratio, crop_height_ratio
    ):
        return coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, SlidingFocusTemplate.WINDOW_WIDTH
        )

    def init_window_generator(
        self, predict_mask, resized_width_ratio, resized_height_ratio
    ):
        width = resized_width_ratio
        height = resized_height_ratio

        shape = Shape(width=width, height=height)
        stride = Stride(
            horizontal=max(int(width / 2), 1),
            vertical=max(int(height / 2), 1),
        )
        increment = Increment(
            width=max(int(width / SlidingFocusTemplate.INCREMENT_FACTOR), 1),
            height=max(int(height / SlidingFocusTemplate.INCREMENT_FACTOR), 1),
        )
        scanner = SlidingWindowScanner(predict_mask, shape, stride)
        self._window_generator = SlidingWindowProcessor(scanner, increment)


class ScannerFocusTemplate(Test2FocusTemplate):

    STRIDE_DIM = (10, 10)

    def init_pareto_front(self):
        self._pareto_front = ParetoFront(solution_dimensions=self._solution_dimensions)

    def generate_resized_ratio(
        self, coordinate_transformer, crop_width_ratio, crop_height_ratio
    ):
        # 100 can be any large number.
        # It is just to ensure that the resized ratio is not 0
        # or that the proportions do not change too much.
        return coordinate_transformer.convert_original_ratio_to_resized(
            crop_width_ratio, crop_height_ratio, 100
        )

    def init_window_generator(
        self, predict_mask, resized_width_ratio, resized_height_ratio
    ):
        width = RESIZED_DIM[0]
        height = int(width * resized_height_ratio / resized_width_ratio)
        if height > RESIZED_DIM[1]:
            height = RESIZED_DIM[1]
            width = int(height * resized_width_ratio / resized_height_ratio)

        shape = Shape(width=width, height=height)
        stride = Stride(
            horizontal=ScannerFocusTemplate.STRIDE_DIM[0],
            vertical=ScannerFocusTemplate.STRIDE_DIM[1],
        )
        self._window_generator = SlidingWindowScanner(predict_mask, shape, stride)
