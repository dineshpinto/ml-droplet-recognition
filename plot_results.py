import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tqdm import tqdm

import image_handler as handler


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


class ModelPlotting:
    def __init__(self, scale_percent: int):
        self.scale_percent = scale_percent

        model_path = os.path.join("models", "droplet_detection_model")
        print(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        time.sleep(2)

        self.training_folder = "training_data"
        images = []
        for file in os.listdir(self.training_folder):
            if file.endswith(".jpg"):
                images.append(file)

        self.images = sorted(images)

    @staticmethod
    def _create_folder(folder: str) -> str:
        print(f"Saving data to {folder}")
        if not os.path.exists(folder):
            os.mkdir(folder)
        return folder

    def plot_model_comparison(self, folder: str):
        print("Saving model comparison images...")
        save_folder = self._create_folder(folder)

        pbar = tqdm(enumerate(self.images))
        for idx, image in pbar:
            pbar.set_description(image)
            img, _ = handler.load_test_image(
                os.path.join(self.training_folder, image),
                circle_label=None,
                scale_percent=self.scale_percent
            )
            pred = self.model.predict(img)

            fig, ax = plt.subplots(ncols=2, sharex="all", sharey="all", figsize=(10, 7))
            ax[0].imshow(img[0, :, :, :], cmap="gray")
            ax[0].set_title("Experimental Data")
            ax[1].imshow(pred[0, :, :, :], cmap="gray")
            ax[1].set_title("Neural Network")

            fig.savefig(os.path.join(save_folder, image), dpi=150, bbox_inches="tight")
            plt.close("all")

    def plot_basic_images(self, folder: str):
        print("Saving basic images...")
        save_folder = self._create_folder(folder)

        pbar = tqdm(enumerate(self.images))
        for idx, image in pbar:
            pbar.set_description(image)
            img, _ = handler.load_test_image(
                os.path.join(self.training_folder, image),
                circle_label=None,
                scale_percent=self.scale_percent
            )
            pred = self.model.predict(img)

            cv2.imwrite(os.path.join(save_folder, image), pred[0, :, :, :] * 255)

    def save_model_layout(self, filepath: str):
        print("Save model layout...")
        print(f"Saving model to {filepath}")
        keras.utils.plot_model(
            self.model,
            to_file=filepath,
            show_shapes=True,
            show_layer_names=True
        )


if __name__ == "__main__":
    mp = ModelPlotting(scale_percent=30)
    mp.plot_model_comparison(os.path.join("output_data", "compare"))
    mp.plot_basic_images(os.path.join("output_data", "basic"))
    mp.save_model_layout(os.path.join("results", "model.png"))
