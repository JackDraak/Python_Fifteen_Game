'''
    This module contains the GUI_Controller class, which is responsible for handling user input and updating the GUI.
'''
# GUI_controller.py 
from Game import Game
import pygame.mixer
import tkinter as tk

pygame.mixer.init()
click_sound = pygame.mixer.Sound("audio/click.wav")
tada_sound = pygame.mixer.Sound("audio/tada.wav")

class Controller:
    def __init__(self, game: Game):
        self.game = game
        self.window = tk.Tk()
        self.window.title("Fifteen Puzzle")
        self.window.geometry("500x530")
        self.window.resizable(False, False)
        self.board = tk.Frame(self.window)
        self.board.pack()
        self.create_tiles()
        self.create_quit_button()
        self.create_volume_slider()
        self.create_win_message()
        self.window.bind("<Button-1>", self.handle_click) # Bind the left mouse button to the handle_click method

    def create_quit_button(self):
        self.quit_frame = tk.Frame(self.window)
        self.quit_frame.pack(pady=10)
        quit_button = tk.Button(self.quit_frame, text="Quit", command=self.window.destroy)
        quit_button.pack()

    def create_tiles(self):
        self.buttons = []
        for row in range(self.game.breadth):
            button_row = []
            for col in range(self.game.breadth):
                label = self.game.get_label(row, col)
                button = tk.Button(self.board, text=label if label != self.game.blank_label else "", width=5, height=2, font=("Arial", 20))
                button.grid(row=row, column=col)
                button_row.append(button)
            self.buttons.append(button_row)
            
    def create_volume_slider(self):
        self.volume_frame = tk.Frame(self.window)
        self.volume_frame.pack(pady=10)
        self.volume_slider = tk.Scale(self.volume_frame, from_=0, to=30, orient=tk.HORIZONTAL, command=self.update_volume, showvalue=False)
        self.volume_var = tk.StringVar()
        self.volume_var.set("Volume: {}%".format(self.scale_to_percentage(self.volume_slider.get())))
        self.volume_label = tk.Label(self.volume_frame, textvariable=self.volume_var)
        self.volume_label.pack(side=tk.LEFT)
        self.volume_slider.pack(side=tk.RIGHT)
        self.volume_slider.set(15)          #  Set the volume slider to 15% by default (relative 50% of max volume)
        
    def create_win_message(self):
        self.win_label = tk.Label(self.board, text="You win!")
        self.win_button = tk.Button(self.board, text="Reshuffle", command=self.handle_reshuffle)
        self.win_label.grid(row=self.game.breadth, column=0, columnspan=self.game.breadth, pady=10)
        self.win_button.grid(row=self.game.breadth+1, column=0, columnspan=self.game.breadth, pady=10)
        self.win_label.grid_remove()        #  Hide the win message initially
        self.win_button.grid_remove()       #  Hide the reshuffle button initially

    def handle_click(self, event) -> None:
        widget = event.widget
        if isinstance(widget, tk.Button):
            tile_row = widget.grid_info()["row"]
            tile_col = widget.grid_info()["column"]
            blank_row, blank_col = self.game.get_position(self.game.blank_label)
            if tile_row == blank_row or tile_col == blank_col:
                if self.game.player_move(self.game.get_label(tile_row, tile_col)):
                    click_sound.play()
            '''
            if (tile_row == blank_row and abs(tile_col - blank_col) == 1) or \
            (tile_col == blank_col and abs(tile_row - blank_row) == 1):
                if self.game.slide_tile(self.game.get_label(tile_row, tile_col)):
                    click_sound.play()
                    # print(game)           #  Utilize Game.__repr__ to see game state after each move
            '''
                    self.update_tiles()
                    if self.game.is_solved():
                        tada_sound.play()
                        self.handle_win()

    def handle_reshuffle(self):
        self.game = Game(self.game.breadth, True)
        self.update_tiles()
        self.win_label.grid_remove()
        self.win_button.grid_remove()
        for button_row in self.buttons:
            for button in button_row:
                if button is not None:
                    button.config(state="normal")

    def handle_win(self):
        self.win_label.grid()
        self.win_button.grid()
        for button_row in self.buttons:
            for button in button_row:
                if button is not None:
                    button.config(state="disabled")
    
    def scale_to_percentage(self, value):
        return int((int(value) / 30) * 100) #  Scale the volume slider value to a percentage (max 30%)

    def update_tiles(self):
        for row in range(self.game.breadth):
            for col in range(self.game.breadth):
                label = self.game.get_label(row, col)
                button = self.buttons[row][col]
                if button is not None:
                    button.config(text=label if label != self.game.blank_label else "")
    
    def update_volume(self, value):
        volume = int(value) / 100
        click_sound.set_volume(volume)
        tada_sound.set_volume(volume)
        self.volume_var.set("Volume: {}%".format(self.scale_to_percentage(value)))

if __name__ == "__main__":
    game = Game(4, True)
    gui = Controller(game)
    gui.window.mainloop()
