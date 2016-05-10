import math
from blessings import Terminal


class Renderer():
    """renders a grid with values"""
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
