import math
import random
import numpy as np
from blessings import Terminal


# used to visualize lines
def make_line(m, b, frm=0, to=200):
    xs = np.linspace(frm, to, 500)
    ys = np.dot(xs[:,np.newaxis], [m]) + b
    return xs, ys


def random_choice(choices):
    """returns a random choice
    from a list of (choice, probability)"""
    # sort by probability
    choices = sorted(choices, key=lambda x:x[1])
    roll = random.random()

    acc_prob = 0
    for choice, prob in choices:
        acc_prob += prob
        if roll <= acc_prob:
            return choice


class Renderer():
    """renders a grid with values (for the gridworld)"""
    cell_width = 7
    cell_height = 3

    def __init__(self, grid):
        self.term = Terminal()
        self.grid = grid

    def _draw_cell(self, x, y, color, value, pos):
        x_mid = math.floor(self.cell_width/2)
        y_mid = math.floor(self.cell_height/2)

        for i in range(self.cell_width):
            for j in range(self.cell_height):
                char = ' '
                print(self.term.move(y+j, x+i) + self.term.on_color(color) + '{}'.format(char) + self.term.normal)

        v = str(value)
        cx = x_mid + x
        cy = y_mid + y
        offset = len(v)//2
        if value < 0:
            highlight = 1
        elif value > 0:
            highlight = 2
        else:
            highlight = 245
        for i, char in enumerate(v):
            x = cx - offset + i
            print(self.term.move(cy, x) + self.term.on_color(color) + self.term.color(highlight) + char + self.term.normal)

        print(self.term.move(y, x) + self.term.on_color(color) + self.term.color(248) + '{},{}'.format(*pos) + self.term.normal)

    def render(self, pos=None):
        """renders the grid,
        highlighting the specified position if there is one"""
        print(self.term.clear())
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val is None:
                    continue
                color = 252 if (r + c) % 2 == 0 else 253
                if pos is not None and pos == (r, c):
                    color = 4
                self._draw_cell(c * self.cell_width, r * self.cell_height, color, val, (r,c))

        # move cursor to end
        print(self.term.move(len(self.grid) * self.cell_height, 0))


class Environment():
    def __init__(self, grid):
        # fill in missing cells
        max_width = max(len(row) for row in grid)
        for row in grid:
            row += [None for _ in range(max_width - len(row))]
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.positions = self._positions()
        self.renderer = Renderer(self.grid)

    def actions(self, pos):
        """possible actions for a state (position)"""
        r, c = pos
        actions = ['stay']
        if r > 0 and self.grid[r-1][c] is not None:
            actions.append('up')
        if r < self.n_rows - 1 and self.grid[r+1][c] is not None:
            actions.append('down')
        if c > 0 and self.grid[r][c-1] is not None:
            actions.append('left')
        if c < self.n_cols - 1 and self.grid[r][c+1] is not None:
            actions.append('right')
        return actions

    def value(self, pos):
        r, c = pos
        return self.grid[r][c]

    def render(self, pos=None):
        self.renderer.render(pos)

    def _positions(self):
        """all valid positions"""
        positions = []
        for r, row in enumerate(self.grid):
            for c, _ in enumerate(row):
                if self.grid[r][c] is not None:
                    positions.append((r,c))
        return positions


class Game():
    def __init__(self, shape=(10,10)):
        self.shape = shape
        self.height, self.width = shape
        self.last_row = self.height - 1
        self.paddle_padding = 1
        self.n_actions = 3 # left, stay, right
        self.term = Terminal()
        self.reset()

    def reset(self):
        # reset grid
        self.grid = np.zeros(self.shape)

        # can only move left or right (or stay)
        # so position is only its horizontal position (col)
        self.pos = np.random.randint(self.paddle_padding, self.width - 1 - self.paddle_padding)
        self.set_paddle(1)

        # item to catch
        self.target = (0, np.random.randint(self.width - 1))
        self.set_position(self.target, 1)

    def move(self, action):
        # clear previous paddle position
        self.set_paddle(0)

        # action is either -1, 0, 1,
        # but comes in as 0, 1, 2, so subtract 1
        action -= 1
        self.pos = min(max(self.pos + action, self.paddle_padding), self.width - 1 - self.paddle_padding)

        # set new paddle position
        self.set_paddle(1)

    def set_paddle(self, val):
        for i in range(1 + self.paddle_padding*2):
            pos = self.pos - self.paddle_padding + i
            self.set_position((self.last_row, pos), val)

    @property
    def state(self):
        return self.grid.reshape((1,-1)).copy()

    def set_position(self, pos, val):
        r, c = pos
        self.grid[r,c] = val

    def update(self):
        r, c = self.target

        self.set_position(self.target, 0)
        self.set_paddle(1) # in case the target is on the paddle
        self.target = (r+1, c)
        self.set_position(self.target, 1)

        # off the map, it's gone
        if r+1 == self.last_row:
            # reward of 1 if collided with paddle, else -1
            if abs(c - self.pos) <= self.paddle_padding:
                return 1
            else:
                return -1
        return 0

    def render(self):
        print(self.term.clear())
        for r, row in enumerate(self.grid):
            for c, on in enumerate(row):
                if on:
                    color = 235
                else:
                    color = 229

                print(self.term.move(r, c) + self.term.on_color(color) + ' ' + self.term.normal)

        # move cursor to end
        print(self.term.move(self.height, 0))


