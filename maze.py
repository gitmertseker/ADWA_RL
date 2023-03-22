import numpy as np
from random import randrange

# If the code is not Cython-compiled, we need to add some imports.
from cython import compiled

# if not compiled:
#     from mazelib.generate.MazeGenAlgo import MazeGenAlgo


import abc
import numpy as np
from numpy.random import shuffle


class MazeGenAlgo:
    __metaclass__ = abc.ABCMeta

    def __init__(self, h, w):
        assert w >= 3 and h >= 3, "Mazes cannot be smaller than 3x3."
        self.h = h
        self.w = w
        self.H = (2 * self.h) + 1
        self.W = (2 * self.w) + 1

    @abc.abstractmethod
    def generate(self):
        return None

    """ All of the methods below this are helper methods,
    common to many maze-generating algorithms.
    """

    def _find_neighbors(self, r, c, grid, is_wall=False):
        """Find all the grid neighbors of the current position; visited, or not.
        Args:
            r (int): row of cell of interest
            c (int): column of cell of interest
            grid (np.array): 2D maze grid
            is_wall (bool): Are we looking for neighbors that are walls, or open cells?
        Returns:
            list: all neighboring cells that match our request
        """
        ns = []

        if r > 1 and grid[r - 2][c] == is_wall:
            ns.append((r - 2, c))
        if r < self.H - 2 and grid[r + 2][c] == is_wall:
            ns.append((r + 2, c))
        if c > 1 and grid[r][c - 2] == is_wall:
            ns.append((r, c - 2))
        if c < self.W - 2 and grid[r][c + 2] == is_wall:
            ns.append((r, c + 2))

        shuffle(ns)
        return ns


class BacktrackingGenerator(MazeGenAlgo):
    """
    1. Randomly choose a starting cell.
    2. Randomly choose a wall at the current cell and open a passage through to any random adjacent
        cell, that has not been visited yet. This is now the current cell.
    3. If all adjacent cells have been visited, back up to the previous and repeat step 2.
    4. Stop when the algorithm has backed all the way up to the starting cell.
    """

    def __init__(self, w, h):
        super(BacktrackingGenerator, self).__init__(w, h)

    def generate(self):
        """highest-level method that implements the maze-generating algorithm
        Returns:
            np.array: returned matrix
        """
        # create empty grid, with walls
        grid = np.empty((self.H, self.W), dtype=np.int8)
        grid.fill(1)

        crow = randrange(1, self.H, 2)
        ccol = randrange(1, self.W, 2)
        track = [(crow, ccol)]
        grid[crow][ccol] = 0

        while track:
            (crow, ccol) = track[-1]
            neighbors = self._find_neighbors(crow, ccol, grid, True)

            if len(neighbors) == 0:
                track = track[:-1]
            else:
                nrow, ncol = neighbors[0]
                grid[nrow][ncol] = 0
                grid[(nrow + crow) // 2][(ncol + ccol) // 2] = 0

                track += [(nrow, ncol)]

        return grid

ins = BacktrackingGenerator(20,20)

maze = ins.generate()
maze = np.delete(maze,-1,0)
maze = np.delete(maze,-1,1)

