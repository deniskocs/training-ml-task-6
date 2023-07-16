import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from deniskocs_python_utils.drawer import Drawer
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image

from train_callback import TrainCallback

print(f"Tensor Flow Version: {tf.__version__}")
print()
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # matplotlib.use('module://backend_interagg')
    print(matplotlib.get_backend())
    matplotlib.use('MacOSX')
    dataset = tfds.load('celeb_a', split='train')

    # dataset = dataset.take(12800)

    # Load the image using PIL
    margo_image = Image.open('../res/margarita.jpg')

    # Convert the image to a NumPy array
    image_array = np.array(margo_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.cast(image_array, tf.float32) / 255.0  # Normalize pixel values
    # plt.imshow(margo_image)
    # plt.show()

    hidden_vector_size = 512

    # Encoder
    encoder_input = keras.layers.Input(shape=(218, 178, 3))
    x = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(encoder_input)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Flatten()(x)
    encoder_output = keras.layers.Dense(hidden_vector_size)(x)
    encoder = keras.Model(encoder_input, encoder_output)

    # Decoder
    decoder_input = keras.layers.Input(shape=(hidden_vector_size))
    x = keras.layers.Dense(28*23*32)(decoder_input)
    x = keras.layers.Reshape((28, 23, 32))(x)
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(3, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.Cropping2D(cropping=((3, 3), (3, 3)))(x)
    decoder_output = x
    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder
    autoencoder_output = decoder(encoder_output)
    autoencoder = keras.Model(encoder_input, autoencoder_output)

    autoencoder.compile(optimizer="Adam", loss="mse")


    def preprocess(example):
        image = example['image']
        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
        return image, image


    new_dataset = dataset.map(preprocess)
    new_dataset = new_dataset.batch(128, num_parallel_calls=tf.data.AUTOTUNE)

    drawer = Drawer()

    train_callback = TrainCallback(drawer=drawer)

    autoencoder.fit(new_dataset, epochs=20) #, callbacks=train_callback)

    # weights_folder = os.getenv('MY_DIRECTORY_PATH')
    # autoencoder.save_weights(weights_folder + "test_weight.h5")

    result = autoencoder.predict(image_array)
    plt.imshow(result[0])
    plt.show()

    # for example in dataset:
    #     print(list(example.keys()))
    #     image = np.array(example["image"])
    #     print(image.shape)
    #     plt.imshow(image)
    #     plt.show()

