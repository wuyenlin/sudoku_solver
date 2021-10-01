#!/usr//bin/python3

import numpy as np

def blk_index(i, j):
    return (i//3) * 3 + j//3


class Solver:
    def __init__(self, puzzle: list) -> np.array:
        self.puzzle = puzzle


    def get_blks(self, puzzle):
        blks = []
        for box_i in range(3):
            for box_j in range(3):
                blk = []
                for i in range(3):
                    for j in range(3):
                        blk.append(puzzle[3*box_i + i][3*box_j + j])
                blks.append(blk)
        return np.array(blks)


    def initialize(self, puzzle):
        """get rows, columns and blocks"""
        self.rows = np.array(puzzle)
        self.cols = np.array(puzzle).T
        self.blks = self.get_blks(puzzle)


    def get_candidates(self, puzzle):
        self.initialize(puzzle)
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


    def fill(self, puzzle):
        # puzzle = self.puzzle.copy()
        self.get_candidates(puzzle)
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


    def match(self, puzzle):
        puzzle = puzzle.copy()
        self.get_candidates(puzzle)
        # Getting the shortest number of candidates > 1:
        min_len = sorted(list(set(map(len, np.array(self.candidates).reshape(1,81)[0]))))[1]
        for i in range(9):
            for j in range(9):
                if len(self.candidates[i][j]) == min_len:
                    for guess in self.candidates[i][j]:
                        puzzle[i][j] = guess
                        solution = self.solve(puzzle)
                        print(solution)
                        if solution is not None:
                            return solution
                        # Discarding a wrong guess
                        puzzle[i][j] = 0


    def solve(self, puzzle):
        self.fill(puzzle)
        if self.correct(puzzle):
            return puzzle
        if not self.valid(puzzle):
            return None
        return self.match(self.puzzle)
    


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
    solution = s.solve(s.puzzle)