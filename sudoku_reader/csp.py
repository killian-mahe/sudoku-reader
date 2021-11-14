# -*- coding: utf-8 -*-
"""CSP representation.

"""
import math
import copy

import numpy as np

from interfaces import Constraint


class CSP:
    """
    A basic implementation of a CSP.

    """

    def __init__(self, variables: list, domains: dict, constraints: list):
        """
        Create a CSP instance.

        Parameters
        ----------
        variables : list
            The list of variables.
        domains : dict
            A dictionary containing the domain of each variable.
        constraints : list
            A list of constraint.

        """
        self.domains = domains
        self.variables = variables
        self.constraints = constraints

        self.var_to_const = {var: set() for var in self.variables}

        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def add_constraints(self, constraint: Constraint):
        """
        Add a new constraint.

        Parameters
        ----------
        constraint : Constraint
            The list of constraints to add.

        Returns
        -------
        None

        """
        for var in constraint.scope:
            if var in self.variables:
                self.var_to_const[var].add(constraint)

    def consistent(self, assignment: dict) -> bool:
        """
        Check if the passed assignment is consistent regarding the CSP.

        Parameters
        ----------
        assignment : dict
            A dictionary {var: domain} representing the assignments of a CSP.

        Returns
        -------
        bool

        """
        return all(
            con.satisfied(assignment)
            for con in self.constraints
            if all(v in assignment for v in con.scope)
        )

    def neighbour(self, var) -> list:
        neighbours = list()
        for constraint in self.var_to_const[var]:
            for other in constraint.scope:
                if other != var:
                    neighbours.append(other)
        return neighbours

    def apply_constraints(self) -> dict:
        assignment = dict()
        for var in self.variables:
            if len(self.domains[var]) == 1:
                assignment[var] = list(self.domains[var])[0]
        return assignment

    def consistent_with(self, assignment: dict, new_assignment: dict) -> bool:
        return self.consistent(assignment | new_assignment)


class SudokuCSP(CSP):
    def __init__(self, sudoku_map: np.ndarray):
        def constraint_evalution(values: any):
            return len(set(values)) == len(values)

        self.sudoku_map = sudoku_map

        variables = list()
        domains = dict()
        constraints = list()

        size = round(math.sqrt(len(sudoku_map)))

        for x in range(len(sudoku_map)):
            for y in range(len(sudoku_map)):
                variables.append(f"{x}, {y}")
                domain = (
                    set(range(1, len(sudoku_map) + 1))
                    if not sudoku_map[x, y]
                    else {sudoku_map[x, y]}
                )
                domains[f"{x}, {y}"] = domain

                for x_row in range(len(sudoku_map)):
                    constraint = Constraint(
                        frozenset({f"{x}, {y}", f"{x_row}, {y}"}), constraint_evalution
                    )
                    if x_row != x and constraint not in constraints:
                        constraints.append(constraint)

                for y_col in range(len(sudoku_map)):
                    constraint = Constraint(
                        frozenset({f"{x}, {y}", f"{x}, {y_col}"}), constraint_evalution
                    )
                    if y_col != y and constraint not in constraints:
                        constraints.append(constraint)

                for i in range(size):
                    for j in range(size):

                        x_box = size * int((x / size)) + i
                        y_box = size * int((y / size)) + j

                        constraint = Constraint(
                            frozenset(
                                {
                                    f"{x}, {y}",
                                    f"{x_box}, {y_box}",
                                }
                            ),
                            constraint_evalution,
                        )

                        if x_box != x and y_box != y and constraint not in constraints:
                            constraints.append(constraint)

        super().__init__(variables, domains, constraints)

    def get_resulted_map(self, assignment: dict) -> np.ndarray:
        """
        Get the resulted map of the CSP.

        Parameters
        ----------
        assignment : dict

        Returns
        -------
        np.ndarray

        """
        result = copy.deepcopy(self.sudoku_map)
        for x in range(0, len(self.sudoku_map)):
            for y in range(0, len(self.sudoku_map)):
                result[x, y] = assignment[f"{x}, {y}"]
        return result
