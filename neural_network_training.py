import os.path
import time

import keras
import keras.layers

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
    # Layer 1 on the network, expand to 16 dimensions and normalize output
    conv1 = keras.layers.Conv2D(filters=16, kernel_size=4, **kwargs)(inputs)
    conv1 = keras.layers.BatchNormalization(momentum=0.99)(conv1)

    # Layer 2 on the network, expand to 32 dimensions and normalize output
    conv2 = keras.layers.Conv2D(filters=32, kernel_size=4, **kwargs)(conv1)
    conv2 = keras.layers.BatchNormalization(momentum=0.99)(conv2)

    # Layer 3 on the network, contract to 16 dimensions and normalize output
    conv3 = keras.layers.Conv2D(filters=16, kernel_size=4, **kwargs)(conv2)
    conv3 = keras.layers.BatchNormalization(momentum=0.99)(conv3)

    # Layer 4 on the network, contract to 1 dimension
    outputs = keras.layers.Conv2D(filters=1, kernel_size=4, **kwargs)(conv3)

    keras_model = keras.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer="Adam", loss="mean_squared_error")

    return keras_model


if __name__ == "__main__":
    DATA_FOLDER = "training_data"
    MODEL_NAME = "droplet_detection_model"
    # Image shape can also be extracted from image itself, currently hard-coded for safety
    IMAGE_SHAPE = (187, 249, 3)

    # Load number of images equal to droplet labels,
    # TODO: Use a dict mapping for labels, current method will work but is error prone
    print(f"Loading {len(droplet_labels)} images from {DATA_FOLDER}...")
    img_batch, label_batch = handler.load_images_from_folder(
        folder=DATA_FOLDER,
        num_images=len(droplet_labels),
        circle_labels=droplet_labels,
        scale_percent=30
    )

    print(f"Loaded image shape = {img_batch.shape}, Label shape = {label_batch.shape}")

    print(f"Generating model")
    model = generate_model(IMAGE_SHAPE)

    t1 = time.time()
    print(f"Starting model fit...")
    model.fit(img_batch, label_batch, batch_size=1000, epochs=1000, verbose=1)
    print(f"Time taken = {int(time.time() - t1)} s")

    model_save_path = os.path.join("models", MODEL_NAME)
    print(f"Saving model to {model_save_path}")
    model.save(model_save_path)
