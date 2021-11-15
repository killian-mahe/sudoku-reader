"""
Picture treatment module.
"""
import numpy as np
import skimage.color, skimage.filters
from scipy import ndimage
from scipy.signal import argrelextrema


def binarize(picture: np.ndarray) -> np.ndarray:
    """
    Binarize the given picture.

    Parameters
    ----------
    picture : np.ndarray
        A N X M color matrix.
    Returns
    -------
    N x M binary matrix.
    """
    width, height, _ = picture.shape
    grey_img = skimage.color.rgb2gray(picture)

    threshold_image = skimage.filters.threshold_sauvola(grey_img, window_size=71, k=0.1)
    return np.where(grey_img < threshold_image, np.ones((width, height)), 0)


def binary_dilatation(picture: np.ndarray) -> np.ndarray:
    """
    Dilate a binary picture.

    Parameters
    ----------
    picture : np.ndarray
        A N x M binary matrix.
    Returns
    -------
    N x M binary matrix.
    """
    return ndimage.binary_dilation(picture).astype(picture.dtype)


def get_largest_connected_components(picture: np.ndarray):
    """
    Get the connected components.

    Parameters
    ----------
    picture : np.ndarray
        A N x M binary matrix.
    Returns
    -------
    N x M binary matrix.
    """
    labeled_img, nb_labels = ndimage.label(picture)

    sizes = ndimage.sum(picture, labeled_img, range(nb_labels + 1))
    mask = sizes == max(sizes)

    filtered_img = mask[labeled_img]
    filtered_img = np.where(filtered_img, filtered_img, 0)
    return np.where(filtered_img == 0, filtered_img, 1)


def get_highest_spikes(picture: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """
    Get the highest spikes over the given axis.

    Parameters
    ----------
    picture : np.ndarray
    n : int
    axis : int

    Returns
    -------
    list
    """
    proj = picture.sum(axis=axis)
    diff = np.diff(proj)

    spikes = argrelextrema(diff, np.greater, order=int(len(diff) * 0.8 * 0.05))[0]

    highest_spikes = spikes[
        np.argpartition(proj[spikes], -min(n, len(spikes)))[-min(n, len(spikes)) :]
    ]
    return np.sort(highest_spikes)


def get_sub_pictures(picture: np.ndarray, rows: list, cols: list):
    """
    Get all the sub sequences of a picture.

    Parameters
    ----------
    picture :  np.ndarray
    rows : list of int
    cols : list of int

    Returns
    -------
    list of np.ndarray
    """
    sub_pictures = list()
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            sub_pictures.append(picture[rows[i] : rows[i + 1], cols[j] : cols[j + 1]])
    return sub_pictures
