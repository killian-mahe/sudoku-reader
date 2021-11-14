# -*- coding: utf-8 -*-
"""Solver algorithms.

"""
import numpy as np

from sudoku_reader.csp import CSP
from sudoku_reader.interfaces import Constraint


def unorder_domain_values(var: any, assignment: dict, csp: CSP):
    """
    Get the domain values of a variable in a random order.

    Parameters
    ----------
    var : any
    assignment : dict
    csp : CSP

    Returns
    -------
    list[any]
    """
    return csp.domains[var]


def random_domain_values(var: any, assignment: dict, csp: CSP):
    """
    Get the domain values of a variable in a random order.

    Parameters
    ----------
    var : any
    assignment : dict
    csp : CSP

    Returns
    -------
    List[any]
    """
    domain = list(csp.domains[var])
    np.random.shuffle(domain)
    return domain


def first_unassigned_variable(assignment: dict, csp: CSP):
    """
    Get the first unselected variable.

    Parameters
    ----------
    assignment : dict
    csp : CSP

    Returns
    -------
    any
    """
    for var in csp.variables:
        if var not in assignment:
            return var


def legal_values_count(csp: CSP, assignment, var):
    related_constraints = csp.var_to_const[var]
    var_domain = csp.domains[var].copy()
    for val in csp.domains[var]:
        for constraint in related_constraints:
            if all(v in assignment for v in constraint.scope):
                if not constraint.satisfied(assignment | {var: val}):
                    var_domain.remove(val)
    return len(var_domain)


def minimum_remaining_value(assignment, csp: CSP):
    min_value_count = 0
    selected_var = None
    for var in csp.variables:
        if var not in assignment:
            legal_values = legal_values_count(csp, assignment, var)
            if not selected_var or legal_values < min_value_count:
                selected_var = var
                min_value_count = legal_values
    return selected_var


def AC3(csp: CSP) -> CSP:
    def remove_inconsistent_values(v, associated_constraint: Constraint) -> bool:
        removed = False
        for value in csp.domains[v].copy():
            for other_var in associated_constraint.scope:
                if other_var != v:
                    violated = True
                    for other_value in csp.domains[other_var]:
                        if associated_constraint.satisfied(
                            {v: value, other_var: other_value}
                        ):
                            violated = False
                            break
                    if violated:
                        csp.domains[v].remove(value)
                        removed = True
        return removed

    arcs = csp.var_to_const.copy()
    while len(arcs) > 0:
        (var, associated_constraints) = arcs.popitem()
        for constraint in associated_constraints:
            if remove_inconsistent_values(var, constraint):
                for affected in csp.neighbour(var):
                    arcs[affected] = csp.var_to_const[affected]

    return csp


def most_constrained_variable(assignment: dict, csp: CSP):
    unassigned_variables = set(csp.variables).symmetric_difference(
        set(assignment.keys())
    )

    unassigned_var_to_const = {
        k: csp.var_to_const[k] for k in unassigned_variables if k in csp.var_to_const
    }

    return sorted(unassigned_var_to_const, key=len)[0]


def least_constraining_value(var: any, assignment: dict, csp: CSP):
    """
    Sort the values of the given variable domain using LCV method.

    Parameters
    ----------
    var : any
    assignment : dict
    csp : CSP

    Returns
    -------
    List[any]
    """
    def conflicts_count(value):
        """
        Count the number of conflicts with the neighbours of this variable.

        Parameters
        ----------
        value : any
            Value of the variable

        Returns
        -------
        int
        """
        values_count = 0
        for constraint in csp.var_to_const[var]:
            for var2 in constraint.scope:
                if var is not var2 and var2 in assignment and not constraint.satisfied(assignment | {var: value}):
                    values_count += 1
        return values_count

    return sorted(csp.domains[var], key=conflicts_count)


def backtracking_search(
    csp: CSP,
    select_unassigned_variable=first_unassigned_variable,
    order_domain_values=unorder_domain_values,
):
    """
    Implementation of the backtracking search algorithm.

    Parameters
    ----------
    csp : CSP
        The constraint satisfaction problem.
    select_unassigned_variable : callable
        How the variables are sorted.
    order_domain_values : callable
        How the domain ise sorted.

    Returns
    -------
    dict
    """
    return recursive_backtracking(
        csp.apply_constraints(), csp, select_unassigned_variable, order_domain_values
    )


def recursive_backtracking(
    assignment: dict,
    csp: CSP,
    select_unassigned_variable=first_unassigned_variable,
    order_domain_values=unorder_domain_values,
):
    """
    Recursive backtracking function.

    Parameters
    ----------
    assignment : dict
        Assignments of variables.
    csp : CSP
        The constraint satisfaction problem.
    select_unassigned_variable : callable
        How the variables are sorted.
    order_domain_values : callable
        How the domain ise sorted.

    Returns
    -------
    dict
    """
    if len(assignment) == len(csp.variables):
        return assignment

    var = select_unassigned_variable(assignment, csp)

    for value in order_domain_values(var, assignment, csp):
        if csp.consistent_with(assignment, {var: value}):
            assignment[var] = value
            result = recursive_backtracking(
                assignment,
                csp,
                select_unassigned_variable=select_unassigned_variable,
                order_domain_values=order_domain_values,
            )
            if result is not None:
                return result
            assignment.pop(var)
    return None
