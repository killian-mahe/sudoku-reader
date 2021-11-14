# -*- coding: utf-8 -*-
"""Generate sudoku puzzles.

"""
import random
from enum import Enum

import numpy as np
import requests

from sudoku_reader.csp import SudokuCSP
from sudoku_reader.algorithms import backtracking_search, random_domain_values


class SudokuDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Generator:

    generator_url = "https://sugoku.herokuapp.com/board"

    @classmethod
    def generate_online(
        cls, size: int = 3, difficulty: SudokuDifficulty = SudokuDifficulty.MEDIUM
    ):
        if size != 3:
            raise NotImplementedError(
                "Sudoku of size different than 3x3 are not currently supported."
            )

        params = {"difficulty": difficulty.value}
        response = requests.get(cls.generator_url, params=params)
        return np.array(response.json()["board"])

    @classmethod
    def generate_backtracking(
        cls, size: int = 3, difficulty: SudokuDifficulty = SudokuDifficulty.MEDIUM
    ):
        sudoku_map = np.zeros((size ** 2, size ** 2), dtype=int)

        csp = SudokuCSP(sudoku_map)

        assignment = backtracking_search(csp, order_domain_values=random_domain_values)
        sudoku_map = csp.get_resulted_map(assignment)

        if difficulty == SudokuDifficulty.EASY:
            i = int(0.5 * (size ** 4))
        elif difficulty == SudokuDifficulty.MEDIUM:
            i = int(0.6 * (size ** 4))
        elif difficulty == SudokuDifficulty.HARD:
            i = int(0.7 * (size ** 4))
        else:
            raise NotImplementedError("You must provide a valid difficulty value.")

        while i:
            x, y = random.randrange(len(sudoku_map)), random.randrange(len(sudoku_map))
            if sudoku_map[y, x] != 0:
                sudoku_map[y, x] = 0
                i -= 1

        return sudoku_map
