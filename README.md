# sudoku_solver
This is a home project that solves the famous puzzle - SUDOKU.


### Example
Input of the puzzle
```
8 2 7 | 5 6 4 | 0 0 0 
0 0 0 | 0 1 0 | 5 0 0 
0 0 1 | 8 0 3 | 0 0 0 
 ---------------------
3 0 0 | 0 4 0 | 9 1 0 
7 0 0 | 2 0 0 | 0 0 8 
0 9 0 | 0 0 1 | 0 6 0 
 ---------------------
0 0 0 | 4 0 0 | 0 0 2 
0 7 0 | 0 3 0 | 0 0 0 
4 0 9 | 0 0 0 | 0 0 0 
```

If there is a unique solution to the input puzzle,
it will return 
```
8 2 7 | 5 6 4 | 1 3 9 
9 4 3 | 7 1 2 | 5 8 6 
6 5 1 | 8 9 3 | 7 2 4 
 ---------------------
3 8 2 | 6 4 7 | 9 1 5 
7 1 6 | 2 5 9 | 3 4 8 
5 9 4 | 3 8 1 | 2 6 7 
 ---------------------
1 3 5 | 4 7 8 | 6 9 2 
2 7 8 | 9 3 6 | 4 5 1 
4 6 9 | 1 2 5 | 8 7 3 
```