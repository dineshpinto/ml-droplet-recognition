import os.path
import time

import tensorflow as tf
from tensorflow import keras
import datetime

import image_handler as handler
from droplet_labels import droplet_labels


def generate_model(image_shape: tuple) -> keras.Model:
    """ Generate a 4 layer neural network. """
    inputs = keras.Input(shape=image_shape)

    kwargs = dict(
        padding='same',
        activation='relu',
        kernel_initializer='glorot_normal',
        kernel_regularizer=None
    )
    # Layer 1 on the network
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=4, **kwargs)(inputs)
    conv1 = keras.layers.BatchNormalization(momentum=0.99)(conv1)

    # Layer 2 on the network
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=4, **kwargs)(conv1)
    conv2 = keras.layers.BatchNormalization(momentum=0.99)(conv2)

    # Layer 3 on the network
    conv3 = keras.layers.Conv2D(filters=16, kernel_size=4, **kwargs)(conv2)
    conv3 = keras.layers.BatchNormalization(momentum=0.99)(conv3)

    # Layer 4 on the network
    outputs = keras.layers.Conv2D(filters=1, kernel_size=4, **kwargs)(conv3)

    keras_model = keras.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer="adam", loss="mean_squared_error")

    return keras_model


if __name__ == "__main__":
    DATA_FOLDER = "training_data"
    MODEL_NAME = "droplet_detection_model"

    # Load number of images equal to droplet labels,
    print(f"Loading {len(droplet_labels)} images from {DATA_FOLDER}...")
    img_batch, label_batch = handler.load_images_from_folder(
        folder=DATA_FOLDER,
        num_images=len(droplet_labels),
        circle_labels=droplet_labels,
        scale_percent=40
    )

    print(f"Loaded image shape = {img_batch.shape}, Label shape = {label_batch.shape}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print(f"Generating model")
    model = generate_model(image_shape=(img_batch.shape[1], img_batch.shape[2], 3))
    print(model.summary())

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    t1 = time.time()
    print(f"Starting model fit...")
    model.fit(
        img_batch,
        label_batch,
        batch_size=1,
        epochs=60,
        verbose=1,
        callbacks=[tensorboard_callback]
    )
    print(f"Time taken to fit = {int(time.time() - t1)} s")

    model_save_path = os.path.join("models", MODEL_NAME)
    print(f"Saving model to {model_save_path}")
    model.save(model_save_path)
