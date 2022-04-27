from typing import Tuple

import cv2
import numpy as np


def create_data_sample(num_shapes: int, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    image = np.zeros((height, width, 3))
    label = np.zeros((height, width, 1))

    for _ in range(num_shapes):
        x = int(np.random.rand() * width)
        y = int(np.random.rand() * height)

        if np.random.rand() > 0.5:
            # Add circle on test and label image
            circle_radius = int(np.ceil(width * 0.15))
            cv2.circle(
                image,
                center=(x, y),
                radius=circle_radius,
                color=np.random.rand(3),
                thickness=cv2.FILLED
            )
            cv2.circle(
                label,
                center=(x, y),
                radius=circle_radius,
                color=(1, 1, 1)
            )
        else:
            # Add rectangle on test image
            rectangle_width = int(np.ceil(width * 0.1 * 0.7))
            start_point = (x - rectangle_width, y - rectangle_width)
            end_point = (x + rectangle_width, y + rectangle_width)
            cv2.rectangle(
                image,
                pt1=start_point,
                pt2=end_point,
                color=np.random.rand(3),
                thickness=cv2.FILLED
            )
    return image, label


def test_batch(num_shapes: int, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    image, label = create_data_sample(num_shapes, height, width)
    return np.expand_dims(image, 0), np.expand_dims(label, 0)
