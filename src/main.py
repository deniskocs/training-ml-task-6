import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset, info = tfds.load('celeb_a', split='train', with_info=True)

    for example in dataset:
        plt.imshow(example["image"])
        plt.show()

