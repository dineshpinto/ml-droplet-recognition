import os
from typing import Tuple, Union

import cv2
import numpy as np


def rescale_image(image: np.ndarray, scale_percent: int) -> np.ndarray:
    """ Reduce resolution of image by a factor of scale_percent. """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_image(
        image_path: str,
        circle_label: tuple,
        overlay: bool = False,
        scale_percent: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This is the main image processor:
        - Load image from filepath
        - Rescale it to reduce resolution
        - Bitwise invert the image
        - Threshold image to increase contrast
        - Cast image as float add overlays if selected
   """
    raw = rescale_image(cv2.imread(image_path), scale_percent=scale_percent)
    _, image = cv2.threshold(cv2.bitwise_not(raw), thresh=200, maxval=255, type=cv2.THRESH_TOZERO_INV)
    image = image.astype("float32") / 255
    label = np.zeros((image.shape[0], image.shape[1], 1), dtype="float32")

    if circle_label:
        x_center = int(np.ceil(circle_label[0] * image.shape[1]))
        y_center = int(np.ceil(circle_label[1] * image.shape[0]))
        radius = int(np.ceil(circle_label[2] * image.shape[1]))

        cv2.circle(
            label,
            center=(x_center, y_center),
            radius=radius,
            color=(1, 1, 1),
            thickness=1
        )

        if overlay:
            cv2.circle(
                image,
                center=(x_center, y_center),
                radius=radius,
                color=(1, 1, 1),
                thickness=1
            )

    return raw, image, label


def load_images_from_folder(
        folder: str,
        num_images: int,
        circle_labels: dict,
        scale_percent: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Load images in a folder until num_images. """
    image_files = []
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            image_files.append(file)

    image_files = sorted(image_files)
    num_labels = len(circle_labels)

    if num_images != num_labels:
        raise ValueError(f"Invalid number of labels ({num_labels}) for images ({num_images})")

    images = []
    labels = []
    for image in image_files[:num_images]:
        _, image, label = load_image(
            image_path=os.path.join(folder, image),
            circle_label=circle_labels[image],
            overlay=False,
            scale_percent=scale_percent
        )
        images.append(image)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def load_test_image(
        image_path: str,
        circle_label: Union[tuple, None],
        scale_percent: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """ Load an image for testing, expands image dims by 1. """
    _, image, label = load_image(image_path, circle_label, overlay=False, scale_percent=scale_percent)
    return np.expand_dims(image, 0), np.expand_dims(label, 0)
