# -*- coding: utf-8 -*-

"""A Sudoku application.

It comes with a GUI that lets you enter your own puzzle or select from a list
of presets and includes a solver that can be invoked interactively (via a
"Solve" button in the GUI) or programmatically (using the ``Sudoku`` class).
"""

from time import perf_counter as t
from itertools import product, islice
from math import sqrt, floor
from random import seed, sample
import multiprocessing as mp
import pickle
from os import cpu_count
from os.path import exists
import tkinter as tk
from tkinter import ttk, PhotoImage
from tkinter.font import Font
from tkinter.messagebox import showerror

import numpy as np

__docformat__ = 'reStructuredText'
__author__ = 'Greg Sotiropoulos <greg.sotiropoulos@gmail.com>'
__version__ = 1, 0, 0
__all__ = 'Sudoku', 'SudokuGames', 'SudokuGui'


class Sudoku:
    """Sudoku puzzle solver.

    The puzzle grid is represented by a 2D numpy array. The solver picks
    an empty square and exhaustively tries all legal values, which for a
    standard 9x9 Sudoku are numbers 1-9 except those that may already appear
    in the same row, column or 3x3 box as the empty square.

    What then determines the runtime of the solver is the algorithm used to
    pick an empty square. The default algorithm is a heuristic one, and one
    that humans intuitively employ (among other heuristics): pick the most
    constrained empty square, ie the square with the smallest set of available
    numbers. The numbers are then tried out one by one, with the algorithm
    backtracking if a particular number in that set does not lead to the
    solution. This simple heuristic turns out to massively speed up the process
    compared to a "brute-force" approach in which an empty square is picked at
    random.

    The class also provides a way for client code to supply alternative
    algorithms for choosing the next candidate square, by setting the
    ``next_fun`` argument to a function -- see the documentation for
    ``solve_recursive()`` and ``solve_iterative()`` for details.
    """

    # Allowed Sudoku sizes, where size means the length of the Sudoku table.
    # By default, only 3 different sizes are supported: 9, 16, 25 --
    # corresponding to box sizes of 3, 4, 5. A standard Sudoku puzzle is a 9x9
    # table that consists of nine 3x3 boxes.
    allowed_sizes = np.arange(3, 6) ** 2

    def __call__(self, *args):
        """Sudoku solution.

        Sudoku instances are callable, solving the puzzle (if it has not
        been solved already) and returning the solution. Note that ``solve``,
        ``solve_iterative`` and ``solve_recursive`` compute the solution but
        store it in ``self.solution`` rather than returning it.

        :return: The solved Sudoku table as a NumPy 2D array.
        """
        if not self.solution:
            self.solve_iterative(*args)
        return self.solution

    @classmethod
    def validate(cls, table):
        """Static method for input validation.

        :param table: An iterable of iterables (eg list of lists) encoding a
        2D integer array or, directly, a 2D NumPy array.

        :raise ValueError: if validation fails.

        :return: a square NumPy array representing the puzzle.
        """
        if table is None:
            raise ValueError('No input table specified.')
        if isinstance(table, cls):
            return table.init_table.copy()
        table = np.array(table, dtype=np.uint8)
        if len(table.shape) != 2:
            raise ValueError('Input table is not 2-dimensional.')
        w, h = table.shape
        if w != h:
            raise ValueError('Input table is not square.')
        bw = floor(sqrt(w))  # box dimensions (3x3 normally)
        if bw**2 != w:
            raise ValueError('Input table width (and height) is not the '
                             'square of an integer.')
        if w not in cls.allowed_sizes:
            raise ValueError(f'Only puzzles of sizes {cls.allowed_sizes} are '
                             f'allowed.')
        return table

    def __init__(self, table, solve=False):
        self.table = table = __class__.validate(table)
        self.init_table = table.copy()
        w = table.shape[0]
        bw = floor(sqrt(w))
        self.range = r = range(w)
        self.alphabet = range(1, w+1)
        self.alphabet_set = set(self.alphabet)

        # For each empty square, store indices of all squares in the
        # row, column and box it belongs to for faster retrieval when solving.
        self.rcb = d = {}
        for i in r:
            for j in r:
                if table[i, j] == 0:
                    sr, sc = i//bw*bw, j//bw*bw
                    # Each item's value is a pair (tuple) of the form
                    # (row1, row2, ...), (col1, col2, ...)
                    # Indices are stored this way so that they can be used
                    # for advanced indexing in Numpy (speed optimization)
                    d[i, j] = *zip(*set().union(
                        product((i,), r),  # indices of i-th row
                        product(r, (j,)),  # indices of j-th column
                        # box indices
                        product(range(sr, sr+bw), range(sc, sc+bw)),
                    ).difference([(i, j)])),  # remove index itself from set

        # Dictionary whose keys are the (row, col) indices of zeros and values
        # are the sets of valid (== legal, as in, not present in the same row,
        # column or box) numbers to choose from.
        self.zeros = dict(
            (idx, self.available_set(idx))
            # index (in (row, col) format) of all empty squares
            for idx in zip(*(table == 0).nonzero())
        )

        self.solution = self.solve_time = None
        if solve:
            self.solve()

    def available_set(self, ij):
        """Available (== allowed, ie not appearing in the same row, column or
        box) numbers at a particular (empty) square.

        :param ij: Location of the square, in (row, col) format.

        :return: Set of available values at the specified location.
        """
        return self.alphabet_set.difference(self.table[self.rcb[ij]])

    def most_constrained_zero(self):
        """Get the coordinates of the empty square with the fewest available
        numbers to choose from, together with a set of those numbers.

        :raise StopIteration: when there are no remaining empty squares.

        :return: A 2-tuple whose first element is the index (row, col) of the
            most constrained zero and second element is the set of available
            values to choose from.
        """
        zs = self.zeros
        try:
            return min(zip(map(len, zs.values()), zs.items()))[1]
        except ValueError:
            raise StopIteration

    # Some alternative implementations of _most_constrained_zero, mostly for
    # benchmarking purposes. Feel free to implement your own method for this,
    # using a different algorithm for choosing which square to try out next!
    def most_constrained_zero_alt1(self):
        zs = self.zeros
        if not zs:
            raise StopIteration
        return min(zip(map(len, zs.values()), zs.items()))[1]

    def most_constrained_zero_alt2(self):
        zs = self.zeros
        if not zs:
            raise StopIteration
        min_key = min(zip(map(len, zs.values()), zs))[1]
        return min_key, zs[min_key]

    def _update_avail_sets(self, ij, new_val):
        """Set current square (the one pointed at by ``ij``) to a new value
        and update the ``self.zeros`` dictionary.

        The dictionary update consists of recomputing the available sets for
        every empty square that is in the same row, column or box as the
        current one.
        """
        table, zs, f = self.table, self.zeros, self.available_set
        table[ij] = new_val  # set the ij-th square to the new value
        # for every square in the same row, column and box
        for idx in zip(*self.rcb[ij]):
            if table[idx] == 0:  # if that square is empty
                zs[idx] = f(idx)  # update the zeros dictionary

    def solve_recursive(self, next_fun=most_constrained_zero):
        """Recursive flavour of the solver (see also ``solve_iterative``).

        The algorithm is essentially the following:
            1. Identify the most 'promising' empty square (by default, the one
               that has the fewest available candidates to pick from)
            2. Pick one of the candidates and assign it to the location of the
               corresponding zero in the 2D array in ``self.table``
            3. Remove that candidate from the ``self.zeros`` dictionary.
            4. Recursively try to solve the new table (which contains one less
               empty square now).
            5. If a solution is found (all squares are full), store it in
               ``self.solution`` and return, otherwise go to step 2 and try
               another candidate.

        :param next_fun: Callable that must return the index of the empty
            square to be considered next, together with the set of numbers that
            are available for that square; these are the numbers 1-9
            (for a 9x9 table), excluding those that already appear in the same
            row, column and smaller (3x3 for the aforementioned table) box.
        """
        t0, table, zs, update_fun = \
            t(), self.table, self.zeros, self._update_avail_sets

        def solve():
            """The core recursive function -- ``solve_recursive`` is just a
            wrapper/initializer.
            """
            ij_min, available = next_fun(self)
            del zs[ij_min]
            while available:
                update_fun(ij_min, available.pop())
                solve()
            update_fun(ij_min, 0)

        try:
            solve()
        # This signals that the self.zeros dictionary is empty, ie there are
        # no more empty squares and thus the solver has succeeded.
        except StopIteration:
            self.solve_time = t() - t0
            self.solution = self.table

    def solve_iterative(self, next_fun=most_constrained_zero):
        """Iterative flavour of ``solve_recursive`` (see its documentation).

        It does not use the call stack (which is a limited resource) but the
        'heap' (pushing to and popping from a list).
        """
        t0, zs, table, update_fun = \
            t(), self.zeros, self.table, self._update_avail_sets

        def getnext():
            """Identical to ``next_fun`` but deletes the returned index from
            the ``self.zeros`` dictionary before returning.

            :return: See documentation for ``most_constrained_zero``
            """
            ij_min_avail = next_fun(self)
            del zs[ij_min_avail[0]]
            return ij_min_avail

        try:
            stack = [getnext()]
            push, pop = stack.append, stack.pop
            while stack:
                ij_min, available = pop()
                while available:
                    update_fun(ij_min, available.pop())
                    push((ij_min, available))
                    ij_min, available = getnext()
                update_fun(ij_min, 0)
        # This signals that the self.zeros dictionary is empty, ie there are
        # no more empty squares and thus the solver has succeeded.
        except StopIteration:
            self.solve_time = t() - t0
            self.solution = table

    def solve(self, solve_fun=solve_iterative, next_fun=most_constrained_zero):
        """Solves the puzzle.

        It calls either the iterative or the recursive flavour of the solver.
        Optionally, a callable for the "next" function can be supplied.

        :param solve_fun: It can be ``Sudoku.solve_iterative`` (the default) or
            ``Sudoku.solve_recursive``. As their name suggests, the only
            difference between them is the flavour of implementation (the
            recursive flavour uses the call stack, the iterative version uses
            the heap (a ``list`` as a stack). The latter should be safer in
            terms of reaching memory limits: the call stack is a limited
            resource and you can potentially hit the interpreter's recursion
            limit and get a ``RecursionError`` (see also
            ``sys.setrecursionlimit()``).

        :param next_fun: callable that selects a currently empty square (see
            documentation for ``most_constrained_zero``).
        """
        solve_fun(self, next_fun)


# GUI

class SudokuGuiEntry(ttk.Entry):
    """Empty square"""

    def __init__(self, *args, sud_grid, table_idx, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_idx = i, j = table_idx
        self.sud_grid = sud_grid
        bw = sud_grid.bw
        r, sr, sc = range(bw**2), i//bw*bw, j//bw*bw
        # indices of entries in the current entry's row, column and box
        self.rcb_idxs = set().union(
            product((i,), r),  # indices of i-th row
            product(r, (j,)),  # indices of j-th column
            product(range(sr, sr+bw), range(sc, sc+bw)),  # box indices
        ).difference((table_idx,))  # excluding the current entry's index

    def get_int(self):
        return int(f'0{self.get()}')

    def is_valid(self, new):
        """Input validation for entries (individual squares).

        Text added/edited by the user is considered valid if it is a positive
        integer that does not appear in the same row, column or (3x3) box as
        the currently edited entry.

        :param new: The value that the entry will have if the editing
            operation is allowed (the "after" value).

        :return: Whether the user edit is a legal number for that square.
        """
        if not new:
            return True
        try:
            grid = self.sud_grid
            available = {*range(1, grid.bw**2 + 1)}.difference(
                grid.entries[idx].get_int() for idx in self.rcb_idxs
            )
            return int(new) in available
        except ValueError:
            return False


class SudokuGuiBox(tk.Frame):
    """3x3 box (Frame grouping together 9 Entry elements)."""

    def __init__(self, sud_grid, bi, bj, *args, **kwargs):
        super().__init__(sud_grid, *args, **kwargs)
        bw = sud_grid.bw
        r = range(bw)
        for i in r:
            for j in r:
                table_idx = bi*bw+i, bj*bw+j
                sud_grid.entries[table_idx] = e = SudokuGuiEntry(
                    self,
                    justify=tk.CENTER,
                    width=0,
                    validate='key',
                    invalidcommand=self.bell,
                    font=Font(size=24),
                    exportselection=0,
                    sud_grid=sud_grid,
                    table_idx=table_idx
                )
                e['validatecommand'] = self.register(e.is_valid), '%P'
                e.grid(row=i, column=j, ipadx=10)


class SudokuGuiGrid(tk.Frame):
    """Full Sudoku grid (Frame grouping together the nine 3x3 boxes)."""

    def __init__(self, parent, tab_or_sz=9, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        if tab_or_sz is None:
            tab_or_sz = 9
        table = Sudoku.validate(
            np.zeros((tab_or_sz,)*2, dtype=np.uint8)
            if isinstance(tab_or_sz, int) else
            tab_or_sz
        )

        self.table_size = sz = table.shape[0]
        self.bw = bw = floor(sqrt(sz))  # side of 3x3 (normally) box
        self.boxes = boxes = {}
        self.entries = {}
        ttk.Style().configure('init.TEntry', foreground='black')
        ttk.Style().configure('user.TEntry', foreground='gray')
        for i, j in product(range(bw), repeat=2):
            boxes[i, j] = b = SudokuGuiBox(self, i, j, borderwidth=2)
            b.grid(row=i, column=j)
        self.loadsud(table)

    def toarray(self, export_styles=False, clear_entries=False):
        entries, w = self.entries, self.bw**2
        table = np.zeros((w, w), dtype=np.uint8)
        r = range(w)
        styles = {}
        for i in r:
            for j in r:
                e = entries[i, j]
                table[i, j] = e.get_int()
                if export_styles:
                    styles[i, j] = e['style']
                if clear_entries:
                    e.delete(0, tk.END)
                    e['style'] = 'user.TEntry'
        if export_styles:
            return table, styles
        return table

    def loadsud(self, table, styles=None):
        table, entries = Sudoku.validate(table), self.entries
        if table.size != len(entries):
            return showerror(
                'Dimension mismatch',
                'Size of Sudoku table does not match the size of the GUI!'
            )

        entry_styles = 'user.TEntry', 'init.TEntry'
        if styles is None:
            styles = {}
        for idx, e in entries.items():
            e.delete(0, tk.END)
            v = table[idx]
            if v:
                e.insert(0, str(v))
            e['style'] = styles.setdefault(idx, entry_styles[min(v, 1)])


class SudokuGuiControls(tk.Frame):
    """All GUI elements below the Sudoku grid."""

    def __init__(self, parent, *args, **kwargs):
        sud17_load_max = kwargs.pop('sud17_load_max', 5000)
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.prev = None  # 1-item "undo" buffer
        pad = parent.outer_pad  # padding of the root frame

        # control (buttons, combobox) specifications start here
        self.load_button = load_btn = ttk.Button(
            self, text='Load:', command=self.load_sample
        )
        load_btn.grid(row=1, column=0, padx=0)

        self.games = SudokuGames(end=sud17_load_max)
        combo_vals = (
            'Easy sample',
            'Medium sample',
            'Hard sample',
            'Extra hard 17-sudoku',
            'Random 17-sudoku'
        )
        self.load_combo = load_cmb = ttk.Combobox(
            self,
            values=combo_vals,
            height=len(combo_vals),
        )
        load_cmb.state(('readonly',))
        load_cmb.grid(row=1, column=1, padx=pad//2)

        self.solve_button = solve_btn = ttk.Button(
            self, text='Solve', command=self.solve
        )
        solve_btn.grid(row=1, column=2, padx=0)

        self.undo_button = un_btn = ttk.Button(
            self, text='Undo', command=self.undo
        )
        un_btn.grid(row=1, column=3, padx=0)

    def load_sample(self):
        """Loads one of the predefined samples (first 3 items) in the ComboBox,
        or a randomly picked 17-sudoku.

        Note that some of the 17-sudoku are dramatically slow to solve in
        comparison to all other sudokus; the former might take anything from
        200 ms to more than a minute to solve, whereas most other puzzles
        (with more than 17 clues, and including the first 3 samples) typically
        take ~5 ms.
        """
        idx = self.load_combo.current()  # index of selected combo value
        if idx >= 0:
            self.clear()
            self.parent.sud_grid.loadsud(
                self.games.samples[idx]  # preset
                if idx < len(self.games.samples) else
                sample(self.games.sudoku17.items(), 1)[0][1] # random 17-sudoku
            )

    def solve(self):
        """Solves the puzzle using the default heuristic (see documentation for
        :class:`Sudoku` for details).
        """
        grid = self.parent.sud_grid
        unsolved, unsolved_styles = grid.toarray(export_styles=True)
        solved = Sudoku(unsolved, solve=True).solution
        if solved is None:
            showerror('Invalid Sudoku', 'This puzzle has no solution!')
        else:
            self.prev = unsolved, unsolved_styles
            grid.loadsud(solved, unsolved_styles)

    def undo(self):
        """Restores the grid to its state just before the last edit."""
        if hasattr(self, 'prev') and self.prev is not None:
            self.parent.sud_grid.loadsud(*self.prev)

    def clear(self):
        """Clears the grid (and saves the existing table for 'undo' purposes"""
        self.prev = self.parent.sud_grid.toarray(
            export_styles=True, clear_entries=True
        )


class SudokuGui(tk.Tk):
    """Sudoku graphical user interface. It is a runnable Tk application."""

    def __init__(self, table_or_size=9, *args, **kwargs):
        """

        :param table_or_size:

        :param args:

        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.title("Sudoku")
        self.resizable(width=False, height=False)
        iconfile = 'sudoku.png'
        if exists(iconfile):
            icon = PhotoImage(master=self, file=iconfile)
            self.tk.call('wm', 'iconphoto', self._w, icon)

        self.outer_pad = pad = 10
        pads = dict(padx=pad, pady=pad)

        # Sudoku grid
        self.sud_grid = g = SudokuGuiGrid(self, table_or_size)
        g.grid(row=0, **pads)

        # Controls (buttons, combobox) below the grid
        self.sud_controls = controls = SudokuGuiControls(self)
        controls.grid(row=1, **pads)

        if table_or_size is None:
            controls.load_combo.current(0)
            controls.load_sample()

        # 1-item buffer for temporary storage of the current table's styles
        self._clip_styles = None

        # keyboard shortcuts
        self.bind_all('<Control-v>', self._paste_from_clip)
        self.bind_all('<Control-c>', self._copy_to_clip)
        self.bind_all('<Control-r>', lambda e: controls.clear())
        self.bind_all('<Control-z>', lambda e: controls.undo())

    def _copy_to_clip(self, event):
        table, self._clip_styles = self.sud_grid.toarray(export_styles=True)
        self.clipboard_clear()
        self.clipboard_append(table.tobytes())

    def _paste_from_clip(self, event):
        clip = self.clipboard_get().encode()
        grid = self.sud_grid
        self.sud_controls.clear()
        grid.loadsud(
            np.frombuffer(clip, dtype=np.uint8).reshape(grid.table_size, -1),
            self._clip_styles
        )


class SudokuGames:
    """Puzzle database and solve scheduler.

    Provides sample puzzles -- 3 of varying levels of difficulty as well as
    an exhaustive collection of puzzles with only 17 clues. It has been
    formally proven that this is the minimum number of clues for a unique
    solution to exist -- any fewer and the puzzle has more than one
    solutions (and is thus not a true puzzle). There are more than 40,000
    17-sudokus.

    The class also provides a bulk solver that uses the ``multiprocessing``
    module to distribute puzzles to a pool of worker processes. This allows
    full utilization of modern multicore (or multi-CPU) hardware to speed
    up this highly CPU-bound task. The bulk solver comes in two flavours,
    synchronous and asynchronous.

    Example:

    >>> from collections import Counter
    >>> sg = SudokuGames(start=10, end=30)
    >>> full_table_cnt = Counter([*range(1, 10)]*9)
    >>> solved = sg.solve_all_17(silent=True)
    >>> all(Counter(table.flatten()) == full_table_cnt for _, table in solved)
    True

    """

    samples = (
        # game 1 - easy
        np.array([
            [0, 6, 0, 1, 0, 4, 0, 5, 0],
            [0, 0, 8, 3, 0, 5, 6, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 1],
            [8, 0, 0, 4, 0, 7, 0, 0, 6],
            [0, 0, 6, 0, 0, 0, 3, 0, 0],
            [7, 0, 0, 9, 0, 1, 0, 0, 4],
            [5, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 7, 2, 0, 6, 9, 0, 0],
            [0, 4, 0, 5, 0, 8, 0, 7, 0]
        ], dtype=np.uint8),

        # game 2: intermediate
        np.array([
            [3, 5, 0, 0, 0, 0, 0, 4, 7],
            [6, 0, 4, 0, 0, 0, 8, 0, 2],
            [0, 0, 0, 8, 0, 1, 0, 0, 0],
            [0, 7, 0, 0, 3, 0, 0, 1, 0],
            [0, 0, 6, 4, 0, 5, 7, 0, 0],
            [0, 8, 0, 0, 2, 0, 0, 3, 0],
            [0, 0, 0, 3, 0, 2, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 2, 0, 5],
            [5, 6, 0, 0, 0, 0, 0, 8, 3]
        ], dtype=np.uint8),

        # game 3: hard(ish)
        np.array([
            [0, 2, 0, 0, 0, 0, 0, 0, 7],
            [0, 7, 0, 0, 0, 4, 0, 1, 0],
            [9, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 8, 0, 6, 3, 0, 0, 0, 2],
            [7, 0, 0, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 1, 8, 0, 6, 0],
            [0, 0, 0, 0, 0, 0, 4, 0, 9],
            [0, 3, 0, 1, 0, 0, 0, 2, 0],
            [4, 0, 0, 0, 0, 0, 0, 8, 0]
        ], dtype=np.uint8),

        # game 4 - one of the hardest 17-sudoku for the automated solver
        np.array([
            [0, 0, 0, 0, 4, 0, 0, 0, 7],
            [0, 0, 3, 5, 0, 0, 0, 0, 0],
            [0, 8, 0, 0, 0, 0, 0, 6, 0],
            [0, 0, 0, 0, 9, 8, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 9, 0, 0, 0, 0, 0, 0, 8],
            [7, 0, 0, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 2, 0, 0]
        ], dtype=np.uint8)
    )

    def __init__(self, start=0, end=None):
        """Loads the 17-sudokus stored in sudoku17.txt into a dictionary for
        later use.

        The text file (downloaded from ) stores 17-sudokus as 81-character
        strings, each on a separate line. These strings are used as keys in a
        dictionary whose values are the NumPy 2D arrays corresponding to that
        string. Optional ``start`` and ``end`` values can be specified in order
        to load a particular range of the 40,000-plus-line text file.

        :param start: Starting line number

        :param end: Ending line number (non-inclusive, as in slice syntax)
        """
        end_str = 'end' if end is None else str(end)
        # name of pickle dump file where solutions are to be stored
        self.solved17_fname = f'sudoku17solved({start},{end_str}).pickle'

        # sudoku17.txt contains all sudoku-17s (puzzles with 17 initially
        # revealed positions). Each line is a single puzzle and consists of a
        # string of 81 digits. When a SudokuGames instance is created, a
        # dictionary of all puzzles, indexed by the 81-digit string, is
        # constructed. The values are the "deserialized" numpy 2D arrays
        # (such as those in SudokuGames.samples).
        with open('sudoku17.txt') as sud17file:
            self.sudoku17 = {
                sud: np.fromiter(sud.rstrip(), dtype=np.uint8).reshape((-1, 9))
                # for each of the first `max_lines` puzzles; `sud` is a
                # 9x9 = 81-digit string
                for sud in islice(sud17file, start, end)
            }

    def solve_all_17(self, silent=False):
        """Solve all loaded 17-sudokus, using ``multiprocessing`` to
        distribute the work to a pool of worker processes.

        :return: A list of (puzzle-string, solved-table) items, identical to
            ``list(self.sudoku17.items())`` but with the solution as the value.
        """
        with mp.Pool() as pool:
            t0 = t()
            items = [*pool.imap_unordered(
                self._get_solution,
                self.sudoku17.items(),
                chunksize=len(self.sudoku17) // cpu_count()
            )]
            with open(self.solved17_fname, 'wb+') as handle:
                pickle.dump(items, handle)
            if not silent:
                print(f'Elapsed time for {len(items)} sudoku17s: '
                      f'{t()-t0:.2f} sec.\n')
            return items

    def solve_all_17_async(self, silent=False):
        """Asynchronous version of ``solve_all_17``"""
        with mp.Pool() as pool:
            self.t0 = t()
            self.silent = silent
            return pool.map_async(
                self._get_solution,
                self.sudoku17.items(),
                chunksize=len(self.sudoku17) // cpu_count(),
                callback=self._save_results
            ).get()

    @staticmethod
    def _get_solution(item):
        # item as the ones in self.sudoku17
        unsolved_str, table = item
        return unsolved_str, Sudoku(table)()

    def _save_results(self, results):
        # ``results`` is a list of 2-tuples, each returned by _get_solution()
        with open(self.solved17_fname, 'wb+') as handle:
            pickle.dump(results, handle)
        if not self.silent:
            print(f'Elapsed time for {len(results)} 17-sudokus: '
                  f'{t()-self.t0:.2f} sec.\n')
        delattr(self, 'silent')


def benchmark():
    """Some basic performance benchmarks. You are encouraged to implement
    your own 'next' function!
    """
    sg = SudokuGames(end=24)
    iter_, recu_ = Sudoku.solve_iterative, Sudoku.solve_recursive

    # tuple of pairs that define the solver function. First element picks the
    # flavour (iterative or recursive), second element picks the 'next'
    # function (see documentation for Sudoku.most_constrained_zero)
    solve_funs = (
        (iter_, Sudoku.most_constrained_zero),
        (iter_, Sudoku.most_constrained_zero_alt1),
        (iter_, Sudoku.most_constrained_zero_alt2)
    )

    seed(0)
    table = sample(sg.sudoku17.items(), 1)[0][1]
    # table = SudokuGames.samples[2]

    print(f'\nSudoku to solve:\n{table}')

    num_runs = 20
    for i, (meth, next_fun) in enumerate(solve_funs):
        sud, tt = None, 0
        for n in range(num_runs):
            sud = Sudoku(table)
            meth(sud, next_fun)
            tt += sud.solve_time
        print(f'\nSolving {num_runs} times with {next_fun.__name__}() '
              f'({meth.__name__.split("_")[1]}): \n'
              f'{tt/num_runs*1000: 0.1f} ms/sudoku\n{sud.solution}')


def main():
    app = SudokuGui()
    app.mainloop()


if __name__ == '__main__':
    # benchmark()
    main()
