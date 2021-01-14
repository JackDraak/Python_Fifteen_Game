import usage


class Tile:
    def __init__(self, label: int, row: int, column: int, dimension: int):
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
        dis = self.distance()
        return f"<Tile> label:{lab}({car}), position:({dim}){row},{col} distance:{dis}"

    def distance(self):
        lab = self.label
        dim = self.dimension
        row = self.row
        col = self.column
        row_dimension = row * dim
        return abs(lab - col - row_dimension - 1)

    def move_to(self, row: int, column: int):
        self.row = row
        self.column = column

    def set(self, cardinal: int, label: int, row: int, column: int):
        self.cardinal = cardinal
        self.label = label
        self.row = row
        self.column = column


if __name__ == '__main__':
    usage.explain()
