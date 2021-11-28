"""
Digits treatment module.
"""
import os

import numpy as np
from scipy import ndimage
import skimage.transform
import keras


MODEL = keras.models.load_model(os.path.join(os.getcwd(), "../model.h5"))


def filter_cells(digits: list):
    """
    Filter each given cell to isolate and center the digit in 28x28 binary array.

    Parameters
    ----------
    digits : array of ((x, y), binary_image)

    Returns
    -------
    An array of the same shape than the given array.
    """
    filtered_digits = digits.copy()

    for i in range(len(digits)):
        img = skimage.transform.resize(digits[i][1], (24, 24), anti_aliasing=False)

        labeled_img, nb_labels = ndimage.label(img)
        sizes = ndimage.sum_labels(img, labeled_img, range(nb_labels + 1))

        digit_label = np.argmax(sizes)
        digit_slice = ndimage.find_objects(labeled_img == digit_label)[0]
        img = img[digit_slice]

        template = np.zeros((28, 28))
        width, height = img.shape
        padding_x = int((28 - width) / 2)
        padding_y = int((28 - height) / 2)

        template[padding_x : padding_x + width, padding_y : padding_y + height] = img

        filtered_digits[i][1] = template

    return filtered_digits


def predict_digit_from_picture(cells: list):
    """
    Predict digits from picture using CNN.

    Parameters
    ----------
    cells

    Returns
    -------

    """
    predicted_digits = list()
    for i in range(len(cells)):
        if len(cells[i][1].shape) < 3:
            cell = np.expand_dims(cells[i][1], 0)
        else:
            cell = cells[i][1]
        prediction = np.argmax(MODEL.predict(cell))
        predicted_digits.append([cells[i][0], prediction])
    return predicted_digits
