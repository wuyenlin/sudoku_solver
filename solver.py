#!/usr/bin/python3


def is_valid(puzzle: list, row, col, cand) -> bool:
    # check for repetition in row and column
    for x in range(9):
        if puzzle[row][x] == cand or puzzle[x][col] == cand:
            return False
 
    # check in subgrid
    starting_row = row - row % 3
    starting_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if puzzle[i + starting_row][j + starting_col] == cand:
                return False
    return True
 

def solve(puzzle, row, col):
    if (row == 9 - 1 and col == 9):
        print("Solved")
        return True

    # proceed to next row
    if col == 9:
        row += 1
        col = 0

    if puzzle[row][col] > 0:
        return solve(puzzle, row, col + 1)

    for cand in range(1, 10): 
        if is_valid(puzzle, row, col, cand):
            puzzle[row][col] = cand
            if solve(puzzle, row, col + 1):
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
        print()
        if i in [2, 5]:
            print("-"*21)


if __name__ == "__main__":
    r1 = [8,0,0,0,1,0,0,0,9]
    r2 = [0,5,0,8,0,7,0,1,0]
    r3 = [0,0,4,0,9,0,7,0,0]
    r4 = [0,6,0,7,0,1,0,2,0]
    r5 = [5,0,8,0,6,0,1,0,7]
    r6 = [0,1,0,5,0,2,0,9,0]
    r7 = [0,0,7,0,4,0,6,0,0]
    r8 = [0,8,0,3,0,9,0,4,0]
    r9 = [3,0,0,0,5,0,0,0,8]
    puzzle = [r1, r2, r3, r4, r5, r6, r7, r8, r9]

    visualize(puzzle)
    if solve(puzzle, 0, 0):
        visualize(puzzle)
    else:
        print("Solution does not exist.")