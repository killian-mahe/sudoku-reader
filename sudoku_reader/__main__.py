"""
Main application program.
"""
import traceback
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal
import numpy as np

from sudoku_reader.interfaces import AlgorithmType, Resolver
from sudoku_reader.gui import MainWindow
from sudoku_reader.csp import SudokuCSP
from sudoku_reader.algorithms import (
    backtracking_search,
    most_constrained_variable,
    minimum_remaining_value,
    least_constraining_value,
    AC3
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
                assignment = backtracking_search(
                    csp
                )

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


if __name__ == "__main__":
    app = QApplication([])

    sudoku_solver = SudokuResolver()
    main_window = MainWindow("Sudoku solver", sudoku_solver)
    main_window.resize(1000, 700)
    main_window.show()

    sys.exit(app.exec())
