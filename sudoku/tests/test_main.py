# -*- coding: utf-8 -*-

import unittest
from sudoku.sudoku import Sudoku, SudokuGames


class TestSudokuValid(unittest.TestCase):
    def setUp(self):
        sg = SudokuGames(end=10)
        # Sample game #3 (hard)
        self.sud_hard_input = table = sg.samples[2]
        self.sud_hard = Sudoku(table, solve=True)

        sud17 = sg.sudoku17
        # First puzzle in the Sudoku-17 list
        self.sud_17_0_input = table = sud17[(*sud17,)[0]]
        self.sud_17_0 = Sudoku(table, solve=True)

    def test_solution_exist(self):
        for sud in (self.sud_hard, self.sud_17_0):
            self.assertIsNotNone(sud.solution)

    def test_solution_consistency(self):
        n = 9
        m = int(n**0.5)
        s = set(range(1, n+1))
        for sud in (self.sud_hard, self.sud_17_0):
            init, solved = sud.init_table, sud.solution
            for i in range(n):
                self.assertCountEqual(solved[i], s)
                self.assertCountEqual(solved[:, i], s)
                i, j = i // m * m, i % m * m
                self.assertCountEqual(solved[i:i+m, j:j+m].flatten(), s)

    def test_unsolved_subset_of_solved(self):
        for sud in (self.sud_hard, self.sud_17_0):
            init, solved = sud.init_table, sud.solution
            filled = init > 0
            self.assertTrue((init[filled] == solved[filled]).all())


if __name__ == '__main__':
    unittest.main()
