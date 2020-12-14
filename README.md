# Sudoku

A, you guessed it, [Sudoku](https://en.wikipedia.org/wiki/Sudoku) application. 
It comes with a graphical user interface (GUI) that lets you enter your own 
puzzle or select from a list of presets. It includes a solver that can be invoked 
interactively (via a _Solve_ button in the GUI) or programmatically.

## Installation
In a terminal/command prompt, type:
```shell
python -m pip install git+https://github.com/GregSotiropoulos/sudoku.git
```

## Usage
To run the GUI application, type:
```commandline
python -m sudoku.sudoku
```
\
To use the module in client code without the GUI, it is sufficient to import `Sudoku`: 
```python
from sudoku.sudoku import Sudoku
```
See the class's documentation for details on what you can do with it.