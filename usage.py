# Playing around with Python 3, continued...
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak

def explain():
    print("""
    Project Name:   'fifteen'
    Project Scope:  The 'fifteen' project is a Python 3 model of the classic 2D tile game known as
    the "15 puzzle". The model and view are contained within the Game class, 'Game.py', which is
    controlled via your preferred controller, i.e. 'GUI_controller.py', 'console_controller.py', or
    'generate_ML_data.py'. Additionally, the Game class relies upon 'Tile.py', the Tile class, used
    as a struct to contain the game-state of the individual tiles of the game matrix.""")


if __name__ == '__main__':
    explain()
