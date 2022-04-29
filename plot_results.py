import os

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
    def __init__(self, scale_percent: int, training_path: str):
        self.scale_percent = scale_percent
        print(f"Set scale percent to {self.scale_percent}")

        model_path = os.path.join("models", "droplet_detection_model")
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)

        self.training_path = training_path
        images = []
        for file in os.listdir(self.training_path):
            if file.endswith(".jpg"):
                images.append(file)

        self.images = sorted(images)

    @staticmethod
    def _create_folder(folder: str) -> str:
        if not os.path.exists(folder):
            print(f"Creating folder {folder}...")
            os.mkdir(folder)
        return folder

    def plot_and_save_model_comparison(self, output_path: str):
        print('_' * 60)
        print(f"Saving model comparison images to {output_path}...")
        output_path = self._create_folder(output_path)

        pbar = tqdm(self.images)
        for idx, image in enumerate(pbar):
            pbar.set_description(image)
            img, _ = handler.load_test_image(
                os.path.join(self.training_path, image),
                circle_label=None,
                scale_percent=self.scale_percent
            )
            pred = self.model.predict(img)

            fig, ax = plt.subplots(ncols=2, sharex="all", sharey="all", figsize=(10, 7))
            ax[0].imshow(img[0, :, :, :], cmap="gray")
            ax[0].set_title("Experimental Data")
            ax[1].imshow(pred[0, :, :, :], cmap="gray")
            ax[1].set_title("Neural Network")

            fig.savefig(os.path.join(output_path, image), dpi=150, bbox_inches="tight")
            plt.close("all")

    def plot_and_save_basic_images(self, output_path: str):
        print('_' * 60)
        print(f"Saving basic images to {output_path}...")
        output_path = self._create_folder(output_path)

        pbar = tqdm(self.images)
        for idx, image in enumerate(pbar):
            pbar.set_description(image)
            img, _ = handler.load_test_image(
                os.path.join(self.training_path, image),
                circle_label=None,
                scale_percent=self.scale_percent
            )
            pred = self.model.predict(img)

            cv2.imwrite(os.path.join(output_path, image), pred[0, :, :, :] * 255)

    def plot_and_save_model_layout(self, output_path: str):
        print('_' * 60)
        print(f"Saving model layout to {output_path}...")
        keras.utils.plot_model(
            self.model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True
        )

    def plot_and_save_overlay(self, input_path: str, output_path: str):
        print('_' * 60)
        print(f"Saving overlay images with input from {input_path} to {output_path}...")
        output_path = self._create_folder(output_path)

        single_circle_detected = 0
        pbar = tqdm(self.images)
        for idx, image in enumerate(pbar):
            pbar.set_description(image)

            img_original, _, _ = handler.load_image(
                os.path.join(self.training_path, image),
                circle_label=None,
                overlay=False,
                scale_percent=self.scale_percent
            )

            img = cv2.cvtColor(cv2.imread(os.path.join(input_path, image)), cv2.COLOR_BGR2GRAY)

            processed = img.copy()
            processed = cv2.GaussianBlur(processed, (5, 5), 0)

            circles = cv2.HoughCircles(
                processed,
                method=cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=30,
                param2=30,
                minRadius=15,
                maxRadius=60
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) == 1:
                    single_circle_detected += 1

                for (x, y, r) in circles:
                    cv2.circle(
                        img_original,
                        center=(x, y),
                        radius=r,
                        color=(255, 255, 0),
                        thickness=4
                    )
                    cv2.rectangle(
                        processed,
                        pt1=(x - 5, y - 5),
                        pt2=(x + 5, y + 5),
                        color=(0, 128, 255),
                        thickness=-1
                    )

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img_original, cmap="gray")
            ax.set_title("Neural Network Output")

            fig.savefig(os.path.join(output_path, image), dpi=150, bbox_inches="tight")
            plt.close("all")

        print(f"Percent single circles detected: {single_circle_detected / len(self.images) * 100:.1f}%")


if __name__ == "__main__":
    mp = ModelPlotting(scale_percent=40, training_path="training_data")

    results_output_path = os.path.join("results", "model.png")
    mp.plot_and_save_model_layout(results_output_path)

    # compare_output_path = os.path.join("output_data", "compare")
    # mp.plot_and_save_model_comparison(compare_output_path)

    basic_output_path = os.path.join("output_data", "basic")
    mp.plot_and_save_basic_images(basic_output_path)

    droplet_overlay_path = os.path.join("output_data", "droplet_detection")
    mp.plot_and_save_overlay(basic_output_path, droplet_overlay_path)
