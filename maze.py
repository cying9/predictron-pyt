"""
Maze generator modified from

https://github.com/brendanator/predictron/blob/master/predictron/maze.py
"""
import numpy as np


class MazeEnv():
    def __init__(self, height=20, width=None, density=0.3, seed=None):
        self.height = height
        self.width = width or height
        self.len = self.height * self.width
        self.np_random = np.random.RandomState()

        # set random seed
        self.seeding(seed=seed)

        # Create the right number of walls to be shuffled for each new maze
        num_locs = self.len - 2
        num_wall = int(num_locs * density)
        self.walls = list('1' * num_wall + '0' * (num_locs - num_wall))

        # Starting point is the bottom right corner
        self.bottom_right_corner = int('0' * (self.len - 1) + '1', 5)

        # Edges for use in flood search
        self.not_left_edge = int(('0' + '1' * (self.width - 1)) * self.height, 2)
        self.not_right_edge = int(('1' * (self.width - 1) + '0') * self.height, 2)
        self.not_top_edge = int('0' * self.width + '1' * self.width * (self.height - 1), 2)
        self.not_bottom_edge = int('1' * self.width * (self.height - 1) + '0' * self.width, 2)

    def gen_emb(self):
        self.np_random.shuffle(self.walls)
        return int('0' + ''.join(self.walls) + '0', base=2)

    def emb2binary(self, emb):
        return bin(emb)[2:].zfill(self.len)

    def emb2maze(self, emb):
        return np.asarray(list(self.emb2binary(emb)),
                          dtype=int).reshape(self.height, self.width)

    def sample_n(self, num_samples):
        return [self.emb2maze(self.gen_emb()) for _ in range(num_samples)]

    def connected_squares(self, emb, start=None):
        """
        Find squares connected to the end square in the
        maze Uses a fast bitwise flood fill algorithm
        """
        empty_squares = ~emb
        azoom = None  # zoom for available nodes
        azoom_next = start or self.bottom_right_corner

        # stop criterion: when all available nodes are found
        while azoom != azoom_next:
            azoom = azoom_next

            # move the zoom (without specific edges) towards all directions
            left = azoom << 1 & self.not_right_edge
            right = azoom >> 1 & self.not_left_edge
            up = azoom << self.width & self.not_bottom_edge
            down = azoom >> self.width & self.not_top_edge

            # combine the available zoom
            azoom_next = (azoom | left | right | up | down) & empty_squares

        return azoom

    def connected_diagonals(self, emb):
        assert self.height == self.width
        return np.diag(self.emb2maze(self.connected_squares(emb)))

    def generate_labelled_mazes(self, num_samples):
        X, Y = [], []
        for _ in range(num_samples):
            x = self.gen_emb()
            X.append(self.emb2maze(x))
            Y.append(self.connected_diagonals(x))

        X = np.asarray(X, dtype=np.float32)[:, np.newaxis, ...]
        Y = np.asarray(Y, dtype=np.float32)
        return X, Y

    def print_maze(self, maze, label):
        """
        Output matrix:
            -A: available nodes;
            -O: available diagonal nodes;
            -X: unavailable nodes.
        """
        out = []
        for n, (row, a) in enumerate(zip(maze, label)):
            row_str = np.where(row.astype(int), 'X', 'A')
            if a == 1:
                row_str[n] = 'O'
            out.append(row_str)
        out = np.asarray(out, dtype=str)
        return out

    def seeding(self, seed=None):
        self.np_random.seed(seed=seed)
        return [seed]
