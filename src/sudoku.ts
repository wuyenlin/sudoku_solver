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

function find_solution(puzzle: number[][], row: number, col: number): boolean{
    if (row==8 && col==9){
        console.log("Solved!");
        return true;
    }

    // proceed to next row
    if (col==9){
        row += 1;
        col = 0;
    }
    if (puzzle[row][col] > 0) return find_solution(puzzle, row, col+1);

    for (var cand=1; cand<10; cand++){
        if (is_valid(puzzle, row, col, cand)){
            puzzle[row][col] = cand;
            if (find_solution(puzzle, row, col+1)) return true;
        }
        // revert to 0 if the above assumption is wrong
        puzzle[row][col] = 0;
    }
    return false;
}


type Puzzle = {r1: string, r2: string, r3: string, r4: string, r5: string,
            r6: string, r7: string, r8: string, r9: string};


class Sudoku {
    private puzzle: HTMLInputElement[];
    private output: HTMLSpanElement;

    constructor(puzzleIds: Puzzle, outputId: string){
        this.puzzle = this.stackElements(puzzleIds);
        this.output = <HTMLSpanElement>document.getElementById(outputId);
        this.wireEvents();
    }

    stackElements(puzzleIds: Puzzle): HTMLInputElement[]{
        var elements: HTMLInputElement[] = [];
        for (var row in puzzleIds) elements.push(<HTMLInputElement>document.getElementById(puzzleIds[row]));
        return elements;
    }
    
    processId(elements: HTMLInputElement[]): number[][]{
        var arrays: number[][] = [];
        for (var i=0; i<10; i++) arrays.push((elements[i].value).split("").map(Number));
        return arrays
    }

    wireEvents(){
        document.getElementById("Solve")!.addEventListener("click",
            event => {
                // this.output.innerHTML = this.Solve(this.processId(this.puzzle)).toString();
                this.output.innerHTML = this.solve(this.processId(this.puzzle));
            });
    }


    solve(puzzle: number[][]): string{
        if (find_solution(puzzle, 0, 0)) {
            console.log(puzzle);
            return "Solved";
        } else{
            return "Unsolved!";
        }
    }
}

window.onload = function () {
    var puzzle: Puzzle = {r1:"r1", r2:"r2", r3:"r3", r4:"r4", r5:"r5",r6:"r6", r7:"r7", r8:"r8", r9:"r9"};
    var solve = new Sudoku(puzzle, "Output");
};
