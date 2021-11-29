"""
Main application program.
"""
import traceback
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, QObject
import matplotlib.pyplot as plt
import numpy as np

from sudoku_reader.interfaces import AlgorithmType, Resolver
from sudoku_reader.gui import MainWindow
from sudoku_reader.csp import SudokuCSP
from sudoku_reader.picture import (
    binarize,
    binary_dilatation,
    get_largest_connected_components,
    get_highest_spikes,
    filter_digit_pictures,
    perspective_transform
)
from sudoku_reader.digits import filter_cells, predict_digit_from_picture
from sudoku_reader.algorithms import (
    backtracking_search,
    most_constrained_variable,
    minimum_remaining_value,
    least_constraining_value,
    AC3,
)


class SudokuResolver(Resolver):
    """
    A worker who manage the resolving of the sudoku.
    """

    result_ready = Signal((AlgorithmType, np.ndarray))
    error = Signal(str)

    def do_work(
        self,
        algorithm_type: AlgorithmType = AlgorithmType.BACKTRACKING,
        sudoku_map: np.array = np.array([]),
    ):
        """
        Do the asked work using the choosen algorithm.

        Parameters
        ----------
        algorithm_type : AlgorithmType
            A type of algorithm to user to resolve the sudoku.
        sudoku_map : np.array
            A array containing the map of the sudoku.

        Returns
        -------
        None
        """
        try:
            algorithm_type = AlgorithmType[algorithm_type.name]

            csp = SudokuCSP(sudoku_map)
            assignment = None

            if algorithm_type is AlgorithmType.BACKTRACKING:
                assignment = backtracking_search(csp)
            elif algorithm_type == AlgorithmType.MRV:
                assignment = backtracking_search(
                    csp, select_unassigned_variable=minimum_remaining_value
                )
            elif algorithm_type == AlgorithmType.DEGREE_H:
                assignment = backtracking_search(
                    csp, select_unassigned_variable=most_constrained_variable
                )
            elif algorithm_type == AlgorithmType.LEAST_CONSTRAINING_H:
                assignment = backtracking_search(
                    csp, order_domain_values=least_constraining_value
                )
            elif algorithm_type == AlgorithmType.AC3:
                csp = AC3(csp)
                assignment = backtracking_search(csp)

            if assignment is not None:
                sudoku_map = csp.get_resulted_map(assignment)
            else:
                self.error.emit(
                    f"Can't find a solution using {algorithm_type.value} algorithm."
                )
        except Exception:
            print(traceback.format_exc())
            self.error.emit(traceback.format_exc())
        self.result_ready.emit(algorithm_type, sudoku_map)


class PictureImporter(QObject):

    result_ready = Signal(list)
    error = Signal(str)

    def do_work(self, picture: np.ndarray):
        try:
            print("Trying to import the picture")
            picture = binarize(picture)

            bin_picture = binary_dilatation(picture)

            bin_picture = perspective_transform(bin_picture)

            grid_picture = get_largest_connected_components(bin_picture)

            x_proj = get_highest_spikes(grid_picture, n=10, axis=0)
            y_proj = get_highest_spikes(grid_picture, n=10, axis=1)

            digits = filter_digit_pictures(bin_picture, y_proj, x_proj)

            digits = filter_cells(digits)
            prediction = predict_digit_from_picture(digits)

            grid = np.zeros((9, 9), dtype=int)
            for (x, y), ch in prediction:
                grid[y, x] = ch

            self.result_ready.emit(grid)

        except Exception:
            print(traceback.format_exc())
            self.error.emit(traceback.format_exc())


if __name__ == "__main__":
    app = QApplication([])

    sudoku_solver = SudokuResolver()
    picture_importer = PictureImporter()
    main_window = MainWindow("Sudoku solver", sudoku_solver, picture_importer)
    main_window.resize(1000, 700)
    main_window.show()

    sys.exit(app.exec())
