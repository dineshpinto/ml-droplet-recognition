import os.path
import time

import keras
import keras.layers

import image_handler as handler
from droplet_labels import droplet_labels


def generate_model(image_shape: tuple) -> keras.Model:
    """ Generate a 4 layer neural network. """
    inputs = keras.Input(shape=image_shape)

    # Layer 1 on the network, expand to 16 dimensions and normalize output
    conv1 = keras.layers.Conv2D(16, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                kernel_regularizer=None)(inputs)
    conv1 = keras.layers.BatchNormalization(momentum=0.99)(conv1)

    # Layer 2 on the network, expand to 32 dimensions and normalize output
    conv2 = keras.layers.Conv2D(32, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                kernel_regularizer=None)(conv1)
    conv2 = keras.layers.BatchNormalization(momentum=0.99)(conv2)

    # Layer 3 on the network, contract to 16 dimensions and normalize output
    conv3 = keras.layers.Conv2D(16, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                kernel_regularizer=None)(conv2)
    conv3 = keras.layers.BatchNormalization(momentum=0.99)(conv3)

    # Layer 4 on the network, contract to 1 dimension
    outputs = keras.layers.Conv2D(1, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                  kernel_regularizer=None)(conv3)

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
    img_batch, label_batch = handler.load_images_from_folder(
        DATA_FOLDER, len(droplet_labels), droplet_labels, scale_percent=30)

    print(f"Image shape = {img_batch.shape}, Label shape = {label_batch.shape}")

    print(f"Generating model")
    model = generate_model(IMAGE_SHAPE)

    t1 = time.time()
    print(f"Starting model fit...")
    model.fit(img_batch, label_batch, batch_size=1000, epochs=1000, verbose=1)
    print(f"Time taken = {int(time.time() - t1)} s")

    model_save_path = os.path.join("models", MODEL_NAME)
    print(f"Saving model to {model_save_path}")
    model.save(model_save_path)
