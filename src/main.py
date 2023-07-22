import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from deniskocs_python_utils.drawer import Drawer
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
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
    validation = tfds.load('celeb_a', split='validation')

    # dataset = dataset.take(12800)

    # Load the image using PIL
    margo_image = Image.open('../res/margarita.jpg')

    # Convert the image to a NumPy array
    image_array = np.array(margo_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.cast(image_array, tf.float32) / 255.0  # Normalize pixel values
    # plt.imshow(margo_image)
    # plt.show()

    hidden_vector_size = 4096
    number_of_samples = 20000

    def make_start_block(hidden_state):
        x = keras.layers.Dense(54 * 44 * 8, activation=tf.keras.activations.relu)(hidden_state)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Reshape((54, 44, 8))(x)
        x = keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=tf.keras.activations.relu)(x)
        x = keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=tf.keras.activations.relu)(x)
        return keras.layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=tf.keras.activations.sigmoid)(x)

    def make_block(hidden_state, image):
        x = keras.layers.Dense(54 * 44 * 8, activation=tf.keras.activations.relu)(hidden_state)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Reshape((54, 44, 8))(x)
        i1 = keras.layers.MaxPool2D(pool_size=4, strides=4)(image)
        x = keras.layers.Concatenate()([i1, x])
        x = keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=tf.keras.activations.relu)(x)
        i2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image)
        x = keras.layers.Concatenate()([i2, x])
        x = keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=tf.keras.activations.relu)(x)
        x = keras.layers.Concatenate()([image, x])
        return keras.layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=tf.keras.activations.sigmoid)(x)

    # 178x218x3
    # 89x109
    decoder_input = keras.layers.Input(shape=(1))
    hidden_state = keras.layers.Embedding(number_of_samples, hidden_vector_size)(decoder_input)
    x = make_start_block(hidden_state)
    x = make_block(hidden_state, x)
    decoder_output = make_block(hidden_state, x)
    decoder = keras.Model(decoder_input, decoder_output)

    print(decoder.summary())

    def ssim_loss(y_true, y_pred):
        return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

    decoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01), loss=tf.keras.losses.mse)

    def preprocess(index, example):
        image = example['image']
        image = tf.image.resize(image, [216, 176])

        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
        return tf.cast(index, tf.float32), image

    indexed_dataset = dataset.take(number_of_samples).enumerate()
    new_dataset = indexed_dataset.map(preprocess)
    new_dataset = new_dataset.shuffle(buffer_size=1024).batch(128, num_parallel_calls=tf.data.AUTOTUNE)

  #  new_validation = validation.map(preprocess).batch(128, num_parallel_calls=tf.data.AUTOTUNE)

    drawer = Drawer()

    train_callback = TrainCallback(drawer=drawer)

    decoder.fit(new_dataset, epochs=1000, callbacks=train_callback)
    # autoencoder.fit(new_dataset, validation_data=new_validation, epochs=20)  # , callbacks=train_callback)
    # weights_folder = os.getenv('MY_DIRECTORY_PATH')
    # autoencoder.save_weights(weights_folder + "test_weight.h5")

    for i in range(100):
        result = decoder.predict([i])
        plt.imshow(result[0])
        plt.show()

    # for example in dataset:
    #     print(list(example.keys()))
    #     image = np.array(example["image"])
    #     print(image.shape)
    #     plt.imshow(image)
    #     plt.show()

