import matplotlib.pyplot as plt

import numpy as np


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
