import time

import keras
import keras.layers

import image_handler as handler
from droplet_labels import droplet_labels


def generate_model(image_shape: tuple) -> keras.Model:
    inputs = keras.Input(shape=image_shape)

    conv1 = keras.layers.Conv2D(16, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                kernel_regularizer=None)(inputs)
    conv1 = keras.layers.BatchNormalization(momentum=0.99)(conv1)

    conv2 = keras.layers.Conv2D(32, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                kernel_regularizer=None)(conv1)
    conv2 = keras.layers.BatchNormalization(momentum=0.99)(conv2)

    conv3 = keras.layers.Conv2D(16, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                kernel_regularizer=None)(conv2)
    conv3 = keras.layers.BatchNormalization(momentum=0.99)(conv3)

    outputs = keras.layers.Conv2D(1, 4, padding='same', activation='relu', kernel_initializer='glorot_normal',
                                  kernel_regularizer=None)(conv3)

    keras_model = keras.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer="Adam", loss="mean_squared_error")

    return keras_model


if __name__ == "__main__":
    img_shape = (187, 249, 3)
    data_folder = "training_data"
    model_save_folder = "droplet_enhancement_model"

    img_batch, label_batch = handler.load_images_from_folder(
        data_folder, len(droplet_labels), droplet_labels, scale_percent=30)

    print(f"Image shape = {img_batch.shape}, Label shape = {label_batch.shape}")

    print(f"Generating model")
    model = generate_model(img_shape)

    t1 = time.time()
    print(f"Starting model fit...")
    model.fit(img_batch, label_batch, batch_size=1000, epochs=1000, verbose=1)
    print(f"Time taken = {int(time.time() - t1)} s")

    print(f"Saving data to {model_save_folder}")
    model.save(model_save_folder)
