import main


class Tile:
    def __init__(self, label, row, column, dimension):
        self.label = label
        self.row = row
        self.column = column
        self.dimension = dimension

    def __repr__(self):
        returns = str(
            f"<Tile> label:{self.label}, position:({self.dimension}){self.row},{self.column} H:{self.h}")
        return returns

    def h(self):
        row_dimension = self.row * self.dimension
        return abs(self.label - self.column - row_dimension - 1)

    def move_to(self, row, column):
        self.row = row
        self.column = column

    def set(self, label, row, column):
        self.label = label
        self.row = row
        self.column = column
