function is_valid(puzzle: number[][], row: number, col: number, cand: number): boolean{
    // check for repetition in row and column
    for (var x=0; x<9; x++){
        if (puzzle[row][x]==cand || puzzle[x][col] == cand) return false;
    }

    // check in subgrid
    var starting_row: number = row - row%3;
    var starting_col: number = col - col%3;

    for (var i=0; i<3; i++){
        for (var j=0; j<3; j++){
            if (puzzle[i+starting_row][j+starting_col] == cand) return false;
        }
    }
    return true;
}

export function find_solution(puzzle: number[][], row: number, col: number): boolean{
    if (row==8 && col==9){
        console.log("Solved!");
        return true;
    }

    // proceed to next row
    if (col==9){
        row += 1;
        col = 0;
    }
    if (puzzle[row][col] > 0) return solve(puzzle, row, col+1);

    for (var cand=1; cand<10; cand++){
        if (is_valid(puzzle, row, col, cand)){
            puzzle[row][col] = cand;
            if (solve(puzzle, row, col+1)) return true;
        }
        // revert to 0 if the above assumption is wrong
        puzzle[row][col] = 0;
    }
    return false;
}
