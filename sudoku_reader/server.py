"""
Flask application.
"""
import json
import os

from flask import Flask, request, abort
import matplotlib.pyplot as plt
import numpy as np

from sudoku_reader.digits import filter_cells, predict_digit_from_picture
from sudoku_reader.picture import (
    binarize,
    binary_dilatation,
    get_largest_connected_components,
    get_highest_spikes,
    filter_digit_pictures
)

app = Flask(__name__)


@app.route('/resolve', methods=['POST'])
def upload_file():

    f = request.files['image']
    filename = "../storage/img.png"
    extension = os.path.splitext(filename)[1]
    print(extension)

    if extension != ".png" and extension != ".jpg" and extension != ".jpeg":
        abort(422)

    f.save(filename)

    picture = plt.imread(filename)
    digits = get_grid(picture)
    print(digits)
    return json.dumps(digits.tolist())


def get_grid(picture: np.array):
    picture = binarize(picture)
    bin_picture = binary_dilatation(picture)

    grid_picture = get_largest_connected_components(bin_picture)

    x_proj = get_highest_spikes(grid_picture, n=10, axis=0)
    y_proj = get_highest_spikes(grid_picture, n=10, axis=1)

    digits = filter_digit_pictures(bin_picture, y_proj, x_proj)

    digits = filter_cells(digits)
    prediction = predict_digit_from_picture(digits)

    grid = np.zeros((9, 9), dtype=int)
    for (x, y), ch in prediction:
        grid[y, x] = ch

    return grid.flatten()
