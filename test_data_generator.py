from typing import Tuple

import cv2
import numpy as np


def add_square(_img: np.ndarray, center: tuple, width: int):
    start_point = (center[0] - width, center[1] - width)
    end_point = (center[0] + width, center[1] + width)
    cv2.rectangle(_img, pt1=start_point, pt2=end_point, color=np.random.rand(3), thickness=cv2.FILLED)


def create_data_sample(num_shapes: int, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    _img = np.zeros((height, width, 3))
    _label = np.zeros((height, width, 1))
    for _ in range(num_shapes):
        x = int(np.random.rand() * width)
        y = int(np.random.rand() * height)
        if np.random.rand() > 0.5:
            cv2.circle(_img, center=(x, y), radius=int(np.ceil(width * 0.15)), color=np.random.rand(3),
                       thickness=cv2.FILLED)
            cv2.circle(_label, center=(x, y), radius=int(np.ceil(width * 0.15)), color=(1, 1, 1))
        else:
            add_square(_img, center=(x, y), width=int(np.ceil(width * 0.1 * 0.7)))
    return _img, _label


def test_batch(num_shapes: int, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    _img, _label = create_data_sample(num_shapes, height, width)
    return np.expand_dims(_img, 0), np.expand_dims(_label, 0)
