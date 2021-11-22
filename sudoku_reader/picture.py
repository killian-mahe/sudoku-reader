"""
Picture treatment module.
"""
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
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
    return ndimage.binary_dilation(picture, iterations=2).astype(picture.dtype)


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


def clear_spikes(binary_picture: np.ndarray, spikes: list, axis: int = 0) -> list:
    """
    Ajust the spike position by checking the max value around the vigen spike.

    Parameters
    ----------
    binary_picture : np.ndarray
    spikes : list
    axis : int

    Returns
    -------
    list of int
    """
    grid_width = spikes[-1] - spikes[0]
    margin = int(grid_width * 0.025)
    proj = binary_picture.sum(axis=axis)
    for i in range(len(spikes)):
        spike = spikes[i]
        lower_bound = max(0, spike - margin)
        upper_bound = min(spike + margin, len(proj))
        spikes[i] = lower_bound + np.argmax(proj[lower_bound:upper_bound])
    return spikes


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

    spikes = clear_spikes(picture, spikes, axis)

    highest_spikes = spikes[
        np.argpartition(proj[spikes], -min(n, len(spikes)))[-min(n, len(spikes)) :]
    ]
    return np.sort(highest_spikes)


def filter_digit_pictures(
    bin_picture: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> list[tuple]:
    """
    Get the digits pictures of a sudoku map.

    Parameters
    ----------
    bin_picture : np.ndarray
        Binarized picture
    rows : np.ndarray
    cols : np.ndarray

    Returns
    -------
    list of tuple
    """
    digits = list()

    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):

            margin = int((cols[j + 1] - cols[j]) * 0.2)

            bin_case = bin_picture[
                rows[i] + margin : rows[i + 1] - margin,
                cols[j] + margin : cols[j + 1] - margin,
            ]

            if np.mean(bin_case.flatten()) > 0.05:
                digits.append(((i, j), bin_case))

    return digits
