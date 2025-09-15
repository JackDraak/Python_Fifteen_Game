# usage.py -- console output for the usage.py snippet explaining the project
def explain():
    print("""Project Name: 'fifteen'
    
    TLDR: Execute Controller (i.e. console_controller.py) of choice to play the game. Execute 
        Game.py directly to run the class unit-tests.
    
    Project Scope:  The 'fifteen' project is a Python 3 model of the classic 2D tile game known
        as the "15 puzzle". The model and view are contained within the Game class, 'Game.py',
        which is controlled via your preferred controller, i.e. 'GUI_controller.py',
        'console_controller.py', or 'AI_controller.py' for RMLA. Additionally, the Game class 
        relies upon the Tile class, used as a struct to contain the game-state of the individual
        tiles of the game matrix.
    """)

if __name__ == '__main__':
    explain()
