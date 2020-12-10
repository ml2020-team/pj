import argparse
import random

import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


# Randomly flip an image.
def flip_left_right(image):
    return tf.image.flip_left_right(image)


# Randomly flip an image.
def flip_up_down(image):
    return tf.image.flip_up_down(image)


# Randomly change an image contrast.
def random_contrast(image1, image2, minval=0.6, maxval=1.4):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image1, image2 = tf.image.adjust_contrast(image1, contrast_factor=r), tf.image.adjust_contrast(image2,
                                                                                                   contrast_factor=r)
    return tf.cast(image1, tf.uint8), tf.cast(image2, tf.uint8)


# Randomly change an image brightness
def random_brightness(image1, image2, minval=0., maxval=.2):
    r = tf.random.uniform([], minval=minval, maxval=maxval)
    image1, image2 = tf.image.adjust_brightness(image1, delta=r), tf.image.adjust_brightness(image2, delta=r)
    return tf.cast(image1, tf.uint8), tf.cast(image2, tf.uint8)


# Randomly change an image saturation
def random_saturation(image1, image2, minval=0.4, maxval=2.):
    r = tf.random.uniform((), minval=minval, maxval=maxval)
    image1, image2 = tf.image.adjust_saturation(image1, saturation_factor=r), tf.image.adjust_saturation(image2,
                                                                                                         saturation_factor=r)
    return tf.cast(image1, tf.uint8), tf.cast(image2, tf.uint8)


# Randomly change an image hue.
def random_hue(image1, image2, minval=-0.04, maxval=0.08):
    r = tf.random.uniform((), minval=minval, maxval=maxval)
    image1, image2 = tf.image.adjust_hue(image1, delta=r), tf.image.adjust_hue(image2, delta=r)
    return tf.cast(image1, tf.uint8), tf.cast(image2, tf.uint8)


# Apply all transformations to an image.
# That is a common image augmentation technique for image datasets, such as ImageNet.
def transform_image(image1, image2):
    if random.randint(1, 3) == 2:
        image1, image2 = image1.transpose((1, 0, 2)), image2.transpose((1, 0, 2))
    if random.randint(1, 3) == 2:
        image1, image2 = flip_up_down(image1), flip_up_down(image2)
    if random.randint(1, 3) == 2:
        image1, image2 = flip_left_right(image1), flip_left_right(image2)

    image1, image2 = random_brightness(image1, image2)
    # image1, image2 = random_hue(image1, image2)
    # image1, image2 = random_saturation(image1, image2)
    # image1, image2 = random_contrast(image1, image2)
    return image1, image2


# Resize transformed image to a 256x256px square image, ready for training.
def resize_image(image):
    image = tf.image.resize(image, size=(args.size, args.size), preserve_aspect_ratio=False)
    image = tf.cast(image, tf.uint8)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img1_path', default='./data/processed_data/type3/temp/pic1.jpg', type=str)
    parser.add_argument('-img2_path', default='./data/processed_data/type3/trgt/pic1.jpg', type=str)
    parser.add_argument('-size', default=100, type=int)
    args = parser.parse_args()

    # Load image to numpy array.
    img1 = PIL.Image.open(args.img1_path)
    img2 = PIL.Image.open(args.img2_path)
    img1.load()
    img2.load()
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Create TensorFlow session.
    session = tf.Session()

    # Display fully pre-processed image.
    transformed_img1, transformed_img2 = transform_image(img1_array, img2_array)
    plt.figure("fully pre-processed image1")
    plt.imshow(PIL.Image.fromarray(transformed_img1.eval(session=session)))
    plt.figure("fully pre-processed image2")
    plt.imshow(PIL.Image.fromarray(transformed_img2.eval(session=session)))

    # Display resized image.
    plt.figure("resized image1")
    plt.imshow(PIL.Image.fromarray(resize_image(transformed_img1).eval(session=session)))
    plt.figure("resized image2")
    plt.imshow(PIL.Image.fromarray(resize_image(transformed_img2).eval(session=session)))
    plt.show()
