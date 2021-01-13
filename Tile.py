# Playing around with Python 3, continued...
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak

class Tile:
    def __init__(self, label, row, column, dimension):
        self.cardinal = label
        self.label = label
        self.row = row
        self.column = column
        self.dimension = dimension

    def __repr__(self):
        lab = self.label
        car = self.cardinal
        dim = self.dimension
        row = self.row
        col = self.column
        row_dimension = row * dim
        h = abs(lab - col - row_dimension - 1)
        return f"<Tile> label:{lab}({car}), position:({dim}){row},{col} H:{h}"

    def move_to(self, row, column):
        self.row = row
        self.column = column

    def set(self, cardinal, label, row, column):
        self.cardinal = cardinal
        self.label = label
        self.row = row
        self.column = column
