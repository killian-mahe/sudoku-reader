"""
Picture treatment module.
"""
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
from scipy import ndimage, spatial
from scipy.signal import argrelextrema
from PIL import Image, ImageDraw, ImageFont
import cv2
import imutils


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


def perspective_transform(binarized_img: np.ndarray) -> np.ndarray:
    binarized_img = binarized_img.astype("uint8")
    cnts = cv2.findContours(
        binarized_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            puzzleCnt = approx.reshape((4, 2)).astype("float32")
            break

    (tl, bl, br, tr) = puzzleCnt
    rect = np.array([tl, tr, br, bl])
    distances = spatial.distance.cdist(rect, rect)

    maxWidth = max(int(distances[0, 1]), int(distances[2, 3]))
    maxHeight = max(int(distances[0, 3]), int(distances[2, 3]))

    margin = 50

    dst = np.array(
        [
            [margin, margin],
            [maxWidth - margin, margin],
            [maxWidth - margin, maxHeight - margin],
            [margin, maxHeight - margin],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(binarized_img, M, (maxWidth, maxHeight))


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
    return ndimage.binary_dilation(picture, iterations=3).astype(picture.dtype)


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

    spikes = argrelextrema(diff, np.greater, order=int(len(diff) * 0.8 * 0.05))[0] + 1

    spikes = clear_spikes(picture, spikes, axis)

    highest_spikes = spikes[
        np.argpartition(proj[spikes], -min(n, len(spikes)))[-min(n, len(spikes)) :]
    ]
    return np.sort(highest_spikes)


def filter_digit_pictures(
    bin_picture: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> list:
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
        margin = int((rows[1] - rows[0]) * 0.15)
        for j in range(len(cols) - 1):

            bin_cell = bin_picture[
                rows[i] + 2 * margin : rows[i + 1] - 2 * margin,
                cols[j] + 2 * margin : cols[j + 1] - 2 * margin,
            ]

            if np.std(bin_cell.flatten()) > 0.3:
                digits.append(
                    [
                        (i, j),
                        bin_picture[
                            rows[i] + margin : rows[i + 1] - margin,
                            cols[j] + margin : cols[j + 1] - margin,
                        ],
                    ]
                )
    return digits


def create_grid_picture(digits: list, file_path: str, size: int = 900):
    """
    Create a sudoku png picture from the given digits list.

    Parameters
    ----------
    digits : list
        Flatten array of digits (empty => 0)
    file_path : str
        Path of the picture.
    size : int, optional
        Size of the picture in pixels.
    """
    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)
    margin = 0.05 * size
    cell_width = int((size - 2 * margin) / 9)
    for x in range(0, 10):
        line_width = 8 if not x % 3 else 2
        draw.line(
            [
                (cell_width * x + margin, margin),
                (cell_width * x + margin, size - margin),
            ],
            fill=(0, 0, 0),
            width=line_width,
        )

    for y in range(0, 10):
        line_width = 8 if not y % 3 else 2
        draw.line(
            [
                (margin, cell_width * y + margin),
                (size - margin, cell_width * y + margin),
            ],
            fill=(0, 0, 0),
            width=line_width,
        )

    font = ImageFont.truetype("arial", int(cell_width * 0.8))

    i = -1
    margin = margin + int(cell_width / 2)
    for y in range(0, 9):
        for x in range(0, 9):
            i += 1
            if digits[i] == 0:
                continue
            draw.text(
                (margin + x * cell_width, margin + y * cell_width),
                str(digits[i]),
                (0, 0, 0),
                align="center",
                font=font,
                anchor="mm",
            )

    img.save(file_path)
