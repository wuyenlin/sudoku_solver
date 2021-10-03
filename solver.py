#!/usr/bin/python3


def solve(puzzle : list, row, col, cand) -> bool:
    # check for repetition
    for x in range(9):
        if puzzle[row][x] == cand or puzzle[x][col] == cand:
            return False
 
    # check subgrid
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if puzzle[i + startRow][j + startCol] == cand:
                return False
    return True
 

def sudoku(puzzle, row, col):
    if (row == 9 - 1 and col == 9):
        print("Solved")
        return True

    # proceed to next row
    if col == 9:
        row += 1
        col = 0

    if puzzle[row][col] > 0:
        return sudoku(puzzle, row, col + 1)

    for cand in range(1, 10): 
        if solve(puzzle, row, col, cand):
            puzzle[row][col] = cand
            if sudoku(puzzle, row, col + 1):
                return True
        # revert assumption
        puzzle[row][col] = 0
    return False
 

def visualize(puzzle: list) -> list:
    for i in range(9):
        for j in range(9):
            if j in [2, 5]:
                print(puzzle[i][j], "|", end=" ")
            else:
                print(puzzle[i][j], end=" ")
        if i in [2, 5]:
            print("\n", "-"*21)
        else:
            print("")


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

    if sudoku(puzzle, 0, 0):
        visualize(puzzle)
    else:
        print("Solution does not exist.")