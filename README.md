# Python: Fifteen Puzzle Game
_The classic 2-dimensional tile-game "Fifteen", modeled in Python 3_

Thank you for your interest in my project. I've been using this repository as a vehicle to get some experience with the Python syntax. Fundamentally, this means it's in flux. It began as a console game, so that I could hit the ground running (i.e. get some solid results, rapidly, without the need to dive into multiple layers of new framework). If you start to dig through the history, your sanity is in your hands -- I take no responsibility. 

At this point, the _console version_ is fairly polished, and offers not only the game "15", but also any matrix dimension from 3-31 (and now, as a bonus, it offers a (q)uit option at every prompt, as well as dual-play modes: WASD or numerical.)

The code is not perfect, but I was also itching to get a _GUI mode_ going, and so now if you run from __GUI.py__ a crude PyGame interface allows the game to be played with a pointer device.

#### Work in Progress:
This seems like a suitable vehicle for a sort of machine-learning 'Hello World!' project.  I won't bore you with the story behind the choice, but I would like to apply one or more forms of pathfinding, artificial intelligence, or machine learning against this game model, for self-edification. If you happen to have found it, may it aid in your own self-edification.



## Files:
  <pre>
  audio                     [folder]  
  images                    [folder]
  console_controller.py     <em>Enhanced Console version of the 15 puzzle.</em>
  Game.py                   Game class: the game model.
  generate_ML_data.py       Produce M.L. training datasets.
  GUI_controller.py         <em>GUI version of the basic 15 puzzle.</em> 
                            (Click the '1' tile on a solved grid to re-shuffle.)
  README.md                 This document.
  Tile.py                   Tile class: struct of tile properties.
  unit_tests.py             Unit tests for Game and Tile classes.
  usage.py                  Generic explainer for non-executable modules.
  </pre>

## Release:
14 Jan 2021: v1.0.0-pre-ML

<https://github.com/JackDraak/Python_Fifteen_Game/releases/tag/v1.0.0-pre-ML>

## Dependencies:
    14 Jan 2021: v1.0.0-pre-ML
      Python interpreter                    (using 3.7.8)
      GUI_controller.py     dep: PyGame     (using 2.0.1)       
      generate_ML_data.py   dep: NumPy      (using 1.19.5)

### Footnote:
    Playing around with Python 3, continued...
    the classic game "Fifteen", for the *console:
    [* now with a PyGame GUI controller]
    (C) 2021 Jack Draak