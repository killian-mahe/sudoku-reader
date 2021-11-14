# -*- coding: utf-8 -*-
"""Application interfaces.

This module regroup all the differents enumeration, interfaces and abstract
class of the application.

"""

from enum import Enum

from PySide6.QtCore import Signal, QObject


class AlgorithmType(Enum):
    BACKTRACKING = "Backtracking"
    MRV = "MRV"
    AC3 = "AC-3"
    DEGREE_H = "Degree heuristic"
    LEAST_CONSTRAINING_H = "Least constraining value"


class Cell:
    """
    Represent a cell in a sudoku puzzle.
    """

    def __init__(self, position: tuple[int], value: int):
        """
        Create an instance of a cell.

        Parameters
        ----------
        position : tuple[int]
            The position of the cell in the sudoku map. Between 0 and len(sudoku_map).
        value : int
            The value of the cell. It must be a number between 0 and 9 included.

        """
        self.position = position
        self._value = round(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: int):
        if value > 9 or value < 0:
            raise RuntimeError("The cell value must be between 0 and 9 included.")
        self._value = round(value)

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, value: int):
        print(
            "Warning: You're trying to change a cell position. You shouldn't have to do it."
        )

        if not isinstance(value, int):
            raise RuntimeError("The position of a cell must be a tuple of integers.")
        self.position = (value, self.position[1])

    @property
    def y(self):
        return self.position[0]

    @y.setter
    def y(self, value: int):
        print(
            "Warning: You're trying to change a cell position. You shouldn't have to do it."
        )

        if not isinstance(value, int):
            raise RuntimeError("The position of a cell must be a tuple of integers.")
        self.position = (self.position[0], value)


class Resolver(QObject):
    """
    A worker who manage the resolving of a problem.
    """

    result_ready = Signal()
    error = Signal()

    def do_work(self):
        self.result_ready.emit()


class Constraint:
    """
    A constraint is composed of a set of variable where the constraint applied
    and a evaluation function.
    """

    def __init__(self, scope: frozenset, val_func: callable):
        """
        Create a Constraint instance.

        Parameters
        ----------
        scope : frozenset
            The list of variables where the constraint is applied.
        val_func : callable
            The evaluation function.
        """
        self.scope = scope
        self.val_func = val_func

    def satisfied(self, assignment: dict):
        """
        Check if the constraint is satisfied in the given assignment.

        Parameters
        ----------
        assignment : dict
            The assignment to check.

        Returns
        -------
        bool
        """
        return self.val_func(tuple(assignment[v] for v in self.scope))

    def __hash__(self):
        return hash((self.scope, self.val_func))

    def __str__(self):
        scope_str = ""
        for v in self.scope:
            scope_str += f"{v}; "
        return scope_str

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            raise RuntimeError(
                "You can only compare a constraint with another constraint."
            )

        return self.scope == other.scope and self.val_func == other.val_func
