#!/bin/bash
import numpy as np

class Solver:
    def __init__(self, all: list) -> np.array:
        self.all = all

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

    def initialize(self):
        """get rows, columns and blocks"""
        self.rows = np.array(self.all)
        self.cols = np.array(self.all).T
        self.blks = self.get_blks(self.all)


    def check_repeat(self, elem):
        if len(elem) != len(set(elem)):
            return False
        return True

    def check_solved(self):
        return True

    def solve(self, all):
        #TODO complete solver ###
        self.initialize(all)
        while True:
            for row in self.rows:
                self.check_repeat(row)
            for col in self.cols:
                self.check_repeat(col)
            for blk in self.blks:
                self.check_repeat(blk)

            if self.check_solved():
                print(solved)
                break
        print("Solved MALAKA")

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
    all = [r1, r2, r3, r4, r5, r6, r7, r8, r9]
    s = Solver(all)
    s.initialize()
    print(s.rows)
    print(s.cols)
    print(s.blks)