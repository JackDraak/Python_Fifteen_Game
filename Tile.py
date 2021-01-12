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
        h = self.h
        return str(f"<Tile> label:{lab}({car}), position:({dim}){row},{col} H:{h}")

    def h(self):
        row_dimension = self.row * self.dimension
        return abs(self.label - self.column - row_dimension - 1)

    def move_to(self, row, column):
        self.row = row
        self.column = column

    def set(self, cardinal, label, row, column):
        self.cardinal = cardinal
        self.label = label
        self.row = row
        self.column = column
