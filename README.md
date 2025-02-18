# Text2Focus

Text2Focus automatically crops images based on textual descriptions, highlighting key areas.

![demo.gif](https://raw.githubusercontent.com/avengerandy/Text2Focus/master/img/demo.gif)

The test images used in this project were captured from a game called Infinity Nikki \[1\].

## 1. Introduction and Features

Text2Focus is a tool designed to automatically crop images, focusing on the most important areas based on both the content of the image and user-defined textual prompts. It is especially useful for tasks that require emphasizing specific features or regions of an image.

### Key Features:

- **Text-Based Focus**: Simply enter a text description (e.g., human face, cartoon character, dog) to guide the algorithm in determining the most relevant regions to focus on. The image is then cropped to highlight these areas based on the user’s input.

- **Automatic Cropping**: After identifying the key areas in the image, the tool automatically performs the crop, ensuring the final output showcases only the most important parts of the image.

- **Multi-Objective Optimization**: The cropping process is not based on a single criterion. The algorithm takes into account several factors (like the number of key points, the proportion of the crop, and how well it captures important features) to determine the best crop that balances all these considerations.

## 2. Algorithm Explanation

### Overview

Text2Focus operates in a two-step process:

1. **Identifying Key Areas**: The algorithm identifies the most important regions of the image by combining the image’s inherent features with the user-provided text description.

2. **Optimizing the Crop**: Once the key areas are identified, the algorithm optimizes the crop, taking multiple factors into account to determine the most balanced and relevant cropped region.

### Identifying Key Areas

The first step of the process involves detecting the most important parts of the image. This is done by combining two models:

- **Pyramid (Saliency Detection)**: Pyramid \[2\]\[3\] identifies areas of the image that are visually prominent. It creates a mask highlighting these key regions.

- **OWLv2 (Text-Conditioned Object Detection)**: OWLv2 \[4\]\[5\] uses the textual prompt provided by the user (e.g., "face," "dog") to detect specific objects or areas in the image that match the description.

Both the saliency mask from Pyramid and the object detection mask from OWLv2 are combined to create a comprehensive mask representing the key regions that should be highlighted in the image.

### Optimizing the Crop

Optimizing the crop is a critical step because it involves balancing multiple factors that contribute to what is considered a "good" crop.

Different users may have different priorities (e.g., more key points, better visual balance, etc.), and a single objective cannot cover all of these preferences effectively.

This is why multi-objective optimization is used—it allows the algorithm to consider several competing factors at once and find the most balanced crop based on the user's needs.

The algorithm evaluates different cropping options using multi-objective optimization, balancing the following objectives:

1. **Total Key Points**: The crop should contain as many of the important regions as possible, ensuring that the most relevant features of the image are preserved.

2. **Proportion of Key Area**: The crop should aim to maximize the proportion of key areas within the selected region. It may conflict with Total Key Points, which could reduce the proportion.

3. **Inclusion of Important Features**: The crop must ensure that significant areas (like faces or objects) are fully captured without cutting them off or leaving them out of the frame.

To optimize the crop, the algorithm evaluates these different objectives using Pareto Front Optimization. This method helps the algorithm find the "best" crop by balancing these different factors.

A Pareto optimal solution means that no crop can improve in one aspect (e.g., more key points) without losing in another (e.g., proportion of key areas or inclusion of important features).

The user can adjust the weight of each factor to select the most suitable solution from the Pareto Front Optimization set, allowing for customized cropping based on their preferences.

This ensures that the final crop is tailored to the user’s priorities, choosing from a set of optimal, non-dominated solutions.

## 3. Implementation Details

### Containers Interaction

The algorithm is divided into three Docker containers based on functionality. Each container performs its respective computation, and communication is carried out through HTTP protocols.

```mermaid
sequenceDiagram

    actor User
    participant main server
    participant Pyramid server
    participant OWLv2 server

    User->>main server: image & prompts
    main server->>Pyramid server: image
    Pyramid server->>main server: Saliency Detection of image
    main server->>OWLv2 server: image & prompts
    OWLv2 server->>main server: Object Detection of image
    activate main server
    Note right of main server: Optimizing the Crop
    deactivate main server
    main server->>User: representative solutions
```

For detailed container setup, please refer to the [docker-compose.yml](https://github.com/avengerandy/Text2Focus/blob/master/docker-compose.yml) file for more information.

### Main Component Details

```mermaid
classDiagram
    namespace sliding_window {
        class Window {
            <<immutable dataclass>>
        }
        class Stride {
            <<immutable dataclass>>
        }
        class Shape {
            <<dataclass>>
        }
        class Increment {
            <<immutable dataclass>>
        }
        class SlidingWindowScanner
        class SlidingWindowProcessor
        class IWindowGenerator
    }

    SlidingWindowScanner o-- Stride
    SlidingWindowScanner o-- Shape
    SlidingWindowScanner o-- Window
    SlidingWindowProcessor o-- Increment
    SlidingWindowProcessor o-- Window
    SlidingWindowProcessor *-- SlidingWindowScanner
    IWindowGenerator <|-- SlidingWindowProcessor
    IWindowGenerator o-- Window
```


```mermaid
classDiagram
    namespace pareto {
        class IParetoFront
        class ParetoFront
        class DynamicRowMatrix
        class Solution
    }

    IParetoFront <|-- ParetoFront
    ParetoFront *-- DynamicRowMatrix
    ParetoFront o-- Solution
```

```mermaid
classDiagram
    class fitness {
        <<namespace>>
        +image_matrix_sum(np.ndarray image_matrix) np.number
        +image_matrix_average(np.ndarray image_matrix) np.number
        +image_matrix_negative_boundary_average(np.ndarray image_matrix) np.number
    }
```

- An overview of what main component inside the containers does and how they contribute to the algorithm's logic.

### Optimization Techniques

#### memory share

In the algorithm, memory is optimized by using references to portions of the matrix, avoiding unnecessary memory duplication.

For example, in the `SlidingWindowScanner`, the `generate_windows` method slices the image matrix instead of creating copies:

```python
class SlidingWindowScanner:
    def generate_windows(self) -> Generator[Window, None, None]:
        for i in range(0, width - window_width + 1, horizontal_stride):
            for j in range(0, height - window_height + 1, vertical_stride):
                sub_image_matrix = self.image_matrix[
                    j : j + window_height, i : i + window_width
                ]
                yield Window(sub_image_matrix=sub_image_matrix, i=i, j=j, ...)
```

Here, sub_image_matrix is just a reference to a section of the original image_matrix, reducing memory overhead by avoiding data duplication.

This reference-based approach also ensures that the SlidingWindowProcessor, which dynamically adjusts the window size and frequently generates new windows, doesn't consume excessive resources when creating multiple matrices.

> Warning: When modifying or extending this code, it's important to remember that many operations in this project rely on shared memory references at the lower level. This practice is common throughout the project, and improper handling could lead to unexpected behavior or excessive memory usage.

#### Optimization Tools

```mermaid
classDiagram
    namespace accelerator {
        class CoordinateTransformer {
            +float scale_x
            +float scale_y
            +convert_resized_to_original()
            +convert_original_to_resized()
            +convert_original_ratio_to_resized()
            +convert_resized_ratio_to_original()
        }
    }
```

```mermaid
classDiagram
    namespace accelerator {
        class DividedParetoFront
    }

    namespace pareto {
        class IParetoFront
        class ParetoFront
    }

    IParetoFront <|-- DividedParetoFront
    DividedParetoFront *-- ParetoFront
```

```mermaid
classDiagram
    namespace accelerator {
        class GeneWindowGenerator
    }

    namespace sliding_window {
        class IWindowGenerator
        class Window
    }
    IWindowGenerator <|-- GeneWindowGenerator
    GeneWindowGenerator o-- Window
```

- Describe any performance optimizations or accelerations that are utilized, such as caching, parallelism, etc.

## 4. Testing & Deployment

- Overview of testing strategies (unit tests, integration tests, etc.)
- Instructions on how to deploy. install and set up the project.

## 5. Usage Guidelines

### Recommendations for how to use the algorithm.

Zero-shot object detection sounds intelligent, but in practical applications, it might just be a fancy feature. The reason for this is that the performance of zero-shot detection is usually lower than algorithms specifically trained to detect certain objects.

There are a few situations users should be aware of:

* **If the object you're detecting is common** (e.g., car, human, etc.), it is highly likely that there is already a pre-existing dataset (e.g., COCO) and pre-trained models available that you can use. In this case, you don't need to rely on OWLv2.

* **If the object you're detecting is uncommon**, but you have image data available, you can consider using OWLv2 as a tool to assist in annotating your data, and then train a specialized model. For example, you can train a model like [Anime Face Detector](https://github.com/qhgz2013/anime-face-detector), a Faster-RCNN-based anime face detector \[6\].

* **Only if you completely lack a dataset**, or if your use case involves unpredictable detection needs (e.g., letting users decide what to detect), should you consider using OWLv2 for zero-shot object detection.

### When to replace components or customize certain parts depending on the use case.

In the algorithm, there are some acceleration tools that you should consider using based on the use case:

#### CoordinateTransformer

- **CoordinateTransformer** is used for scaling images. This is almost mandatory because larger images consume more resources. Always use this to ensure the image is properly scaled before processing.

#### Crop Mode

![crop_mode.png](https://raw.githubusercontent.com/avengerandy/Text2Focus/master/img/crop_mode.png)

- **Precise Crop Mode** involves using `SlidingWindowProcessor`, a brute-force method that scans all possible solutions. This results in many potential solutions, so it's recommended to use `DividedParetoFront` for acceleration. This helps speed up the processing without losing accuracy. You can check `example_server_sliding_processor.py` for more details.

- **Approximation Crop Mode** uses `GeneWindowGenerator` to generate approximate solutions within a fixed time frame. Since the genetic algorithm produces fewer solutions, `ParetoFront` can be used directly without the need for additional acceleration. You can refer to `example_server_gene_window.py` for how this is implemented.

#### Fill Mode

![full_mode.png](https://raw.githubusercontent.com/avengerandy/Text2Focus/master/img/full_mode.png)

- **Fill Mode** is similar to CSS's `cover`, which stretches the image to cover the entire container while maintaining the aspect ratio. In this case, you can use `SlidingWindowScanner`. There's no need for multi-objective optimization here since the `cover` mode inherently stretches the image to its maximum, ensuring the image fits the container. You can refer to `example_server_sliding_scanner.py` for implementation details.

#### Conclusion

| Mode | IWindowGenerator | IParetoFront |
|------|------------------|--------------|
| Precise Crop Mode (Slower) | SlidingWindowProcessor | DividedParetoFront |
| Approximation Crop Mode (Faster) | GeneWindowGenerator | ParetoFront |
| Fill Mode (Special Case) | SlidingWindowScanner | None |

## 6. License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/avengerandy/Text2Focus/blob/master/LICENSE) file for more details.

## 7. References

[1] Infold Games. Infinity Nikki Official Website – The Coziest Open-World Game. Infinity Nikki Official Website, n.d., https://infinitynikki.infoldgames.com/en/home.

[2] Zhao, Ting, and Xiangqian Wu. "Pyramid feature attention network for saliency detection." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

[3] Sairajk. PyTorch Pyramid Feature Attention Network for Saliency Detection. GitHub, n.d., https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection.

[4] Minderer, Matthias, Alexey Gritsenko, and Neil Houlsby. "Scaling open-vocabulary object detection." Advances in Neural Information Processing Systems 36 (2024).

[5] Google. OWL-ViT 2 Base Patch 16 Ensemble. Hugging Face, n.d., https://huggingface.co/google/owlv2-base-patch16-ensemble.

[6] Qhgz2013. Anime Face Detector: A Faster-RCNN based anime face detector. GitHub, n.d., https://github.com/qhgz2013/anime-face-detector.
