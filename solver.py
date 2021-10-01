#!/usr//bin/python3

import numpy as np

def blk_index(i, j):
    return (i//3) * 3 + j//3


class Solver:
    def __init__(self, all: list) -> np.array:
        self.original = all


    def get_blks(self, all):
        blks = []
        for box_i in range(3):
            for box_j in range(3):
                blk = []
                for i in range(3):
                    for j in range(3):
                        blk.append(all[3*box_i + i][3*box_j + j])
                blks.append(blk)
        return np.array(blks)


    def initialize(self, all):
        """get rows, columns and blocks"""
        self.rows = np.array(all)
        self.cols = np.array(all).T
        self.blks = self.get_blks(all)


    def get_candidates(self, all):
        self.initialize(all)
        candidates = []
        for i in range(9):
            row_candidates = []
            for j in range(9):
                row = set(self.rows[i])
                col = set(self.cols[j])
                sub = set(self.blks[blk_index(i, j)])

                common = row | col | sub
                cand = set(range(10)) - common

                if not self.rows[i][j]:
                    row_candidates.append(list(cand))
                else:
                    row_candidates.append([self.rows[i][j]])
            candidates.append(row_candidates)
        self.candidates = candidates


    def solve(self):
        puzzle = self.original.copy()
        self.get_candidates(puzzle)
        print(self.candidates)
        to_fill = True
        while to_fill:
            to_fill = False
            for i in range(9):
                for j in range(9):
                    if len(self.candidates[i][j]) == 1 and puzzle[i][j] == 0:
                        puzzle[i][j] = self.candidates[i][j][0]
                        self.get_candidates(puzzle)
                        to_fill = True
        self.puzzle = puzzle
    

    def correct(self, puzzle):
        """ a boolean function that determines if the puzzle is solved"""
        if np.all(np.sum(puzzle, axis=1) == 45) and \
            np.all(np.sum(puzzle, axis=0) == 45) and \
            np.all(np.sum(self.get_blks(puzzle), axis=1) == 45):
            return True
        return False


    def valid(self, puzzle):
        self.get_candidates(puzzle)
        for i in range(9):
            for j in range(9):
                if len(self.candidates[i][j]) == 0:
                    return False
        return True


    def filter(self):
        from copy import deepcopy
        # Check for empty cells
        test = self.original.copy()
        self.get_candidates(test)
        filtered_candidates = deepcopy(self.candidates)
        for i in range(9):
            for j in range(9):
                if test[i][j] == 0:
                    for candidate in self.candidates[i][j]:
                        # Use test candidate
                        test[i][j] = candidate
                        # Remove candidate if it produces an invalid grid
                        if not is_valid_grid(self.solve(test)):
                            filtered_candidates[i][j].remove(candidate)
                        # Revert changes
                        test[i][j] = 0


    def main(self):
        self.solve()


if __name__ == "__main__":
    r1 = [0,0,0,5,0,4,0,2,0]
    r2 = [0,3,0,2,0,6,8,0,0]
    r3 = [0,1,0,0,9,0,0,4,0]
    r4 = [0,2,4,0,7,0,0,0,0]
    r5 = [0,0,6,0,0,0,1,0,5]
    r6 = [3,0,9,0,0,0,0,0,0]
    r7 = [0,0,3,4,0,0,0,0,0]
    r8 = [0,0,0,0,5,7,0,0,0]
    r9 = [0,4,8,1,6,3,0,0,0]
    puzzle = [r1, r2, r3, r4, r5, r6, r7, r8, r9]
    s = Solver(puzzle)
    s.main()