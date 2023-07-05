import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt

from drawer import Drawer
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

    # Encoder
    encoder_input = keras.layers.Input(shape=(218, 178, 3))
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                activation=keras.activations.relu)(encoder_input)  # (None, 108, 88, 32)
    encoder_output = conv1
    print(conv1.shape)
    encoder = keras.Model(encoder_input, encoder_output)

    # Decoder
    decoder_input = keras.layers.Input(shape=(109, 89, 32))
    upscale1 = keras.layers.UpSampling2D()(decoder_input)
    print(upscale1.shape)
    conv1 = keras.layers.Convolution2D(filters=3, kernel_size=(3, 3), padding="same",
                                       activation=keras.activations.sigmoid)(upscale1)
    print(conv1.shape)
    decoder_output = conv1
    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder
    autoencoder_output = decoder(encoder_output)
    autoencoder = keras.Model(encoder_input, autoencoder_output)

    autoencoder.compile(optimizer="Adam", loss="mse")


    def preprocess(example):
        image = example['image']
        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
        return image, image


    new_dataset = dataset.take(100).map(preprocess)
    new_dataset = new_dataset.batch(32, num_parallel_calls=tf.data.AUTOTUNE)

    drawer = Drawer()

    train_callback = TrainCallback(drawer=drawer)

    autoencoder.fit(new_dataset, epochs=32, callbacks=train_callback)

    for example in dataset:
        print(list(example.keys()))
        image = np.array(example["image"])
        print(image.shape)
        plt.imshow(image)
        plt.show()
