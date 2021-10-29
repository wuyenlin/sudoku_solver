function repeat_string(string: string, times: number): string{
    var repeated: string = "";
    while (times > 0){
        repeated += string;
        times--;
    }
    return repeated
}


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

function solve(puzzle: number[][], row: number, col: number): boolean{
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

function visualize(puzzle: number[][]): void{
    for (var i=0; i<9; i++){
        var row: number[] = [];
        for (var j=0; j<9; j++) row.push(puzzle[i][j]);
        console.log(row);
        if (i==2 || i==5){
            var h_grid: string = repeat_string("-", 23);
            console.log(h_grid);
        }
    }
}


// Execute an example
let r1: number[] = [8,0,0,0,1,0,0,0,9];
let r2: number[] = [0,5,0,8,0,7,0,1,0];
let r3: number[] = [0,0,4,0,9,0,7,0,0];
let r4: number[] = [0,6,0,7,0,1,0,2,0];
let r5: number[] = [5,0,8,0,6,0,1,0,7];
let r6: number[] = [0,1,0,5,0,2,0,9,0];
let r7: number[] = [0,0,7,0,4,0,6,0,0];
let r8: number[] = [0,8,0,3,0,9,0,4,0];
let r9: number[] = [3,0,0,0,5,0,0,0,8];
let puzzle: number[][] = [r1,r2,r3,r4,r5,r6,r7,r8,r9];

visualize(puzzle);
if (solve(puzzle, 0, 0)){
    visualize(puzzle);
} else{
    console.log("Solution does not exist.");
}
