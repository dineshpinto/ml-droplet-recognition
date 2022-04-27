import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
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
    _, image = cv2.threshold(cv2.bitwise_not(raw), 200, 255, cv2.THRESH_TOZERO_INV)
    image = image.astype("float32") / 255

    label = np.zeros((image.shape[0], image.shape[1], 1), dtype="float32")
    if circle_label:
        cv2.circle(
            label,
            center=(circle_label[0], circle_label[1]),
            radius=int(np.ceil(label.shape[1] * circle_label[2])),
            color=(1, 1, 1),
            thickness=2
        )

    if circle_label and overlay:
        cv2.circle(
            image,
            center=(circle_label[0], circle_label[1]),
            radius=int(np.ceil(label.shape[1] * circle_label[2])),
            color=(1, 1, 1),
            thickness=2
        )

    return raw, image, label


def load_images_from_folder(
        folder: str,
        num_images: int,
        circle_labels: list,
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
    for idx, image in enumerate(image_files[:num_images]):
        _, image, label = load_image(
            image_path=os.path.join(folder, image),
            circle_label=circle_labels[idx],
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
        circle_label: tuple,
        scale_percent: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """ Load an image for testing, expands image dims by 1. """
    _, image, label = load_image(image_path, circle_label, overlay=False, scale_percent=scale_percent)
    return np.expand_dims(image, 0), np.expand_dims(label, 0)


def to_three_channels(label: np.ndarray) -> np.ndarray:
    """ Cast a 1 channel image to a 3 channel image """
    return np.squeeze(np.stack((label, label, label), axis=2))


def show_sample(image: np.ndarray, label: np.ndarray) -> plt.Figure:
    """ Plot a single image. """
    fig, ax = plt.subplots(ncols=2, sharex="all", sharey="all", figsize=(10, 7))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Raw Data")
    ax[1].imshow(to_three_channels(label), cmap="gray")
    ax[1].set_title("Labelled Data")
    return fig


def show_batch(images: np.ndarray, labels: np.ndarray, idx: int = 0) -> plt.Figure:
    """ Plot a single image at index `idx` from an image array. """
    return show_sample(images[idx, :, :, :], labels[idx, :, :, :])


def show_result(images: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> plt.Figure:
    """ Plot the result of a training run. """
    fig, ax = plt.subplots(ncols=3, sharex="all", sharey="all", figsize=(10, 7))
    ax[0].imshow(images[0, :, :, :], cmap="gray")
    ax[0].set_title("Raw Data")
    ax[1].imshow(to_three_channels(labels[0, :, :, :]), cmap="gray")
    ax[1].set_title("Labelled Data")
    ax[2].imshow(to_three_channels(predictions[0, :, :, :]), cmap="gray")
    ax[2].set_title("Neural Network")
    return fig
