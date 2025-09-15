# ncurses_controller.py
"""
NCurses-based controller for the sliding puzzle game.
Provides a visual interface with cursor navigation and tile selection.
"""

import curses
from Game import Game
from console_controller import Controller
from typing import Tuple, Optional


class NCursesController:
    def __init__(self, game: Game):
        self.game = game
        self.cursor_row = 0
        self.cursor_col = 0
        self.move_count = 0
        self.stdscr = None
        self.quit_completely = False
        
        # Color pairs (will be initialized in setup_colors)
        self.COLOR_NORMAL = 1      # Normal tiles
        self.COLOR_SELECTED = 2    # Currently selected tile
        self.COLOR_BLANK = 3       # Blank space
        self.COLOR_BORDER = 4      # Game border
        self.COLOR_STATUS = 5      # Status text
        self.COLOR_ERROR = 6       # Error messages
        self.COLOR_SUCCESS = 7     # Success messages

    def setup_colors(self):
        """Initialize color pairs for the interface."""
        if not curses.has_colors():
            return
            
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(self.COLOR_NORMAL, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(self.COLOR_SELECTED, curses.COLOR_YELLOW, curses.COLOR_BLUE) 
        curses.init_pair(self.COLOR_BLANK, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(self.COLOR_BORDER, curses.COLOR_WHITE, curses.COLOR_MAGENTA)
        curses.init_pair(self.COLOR_STATUS, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_SUCCESS, curses.COLOR_YELLOW, curses.COLOR_GREEN)

    def draw_tile(self, row: int, col: int, tile_row: int, tile_col: int):
        """Draw a single tile at the specified screen position."""
        label = self.game.get_label(tile_row, tile_col)
        is_blank = (label == self.game.blank_label)
        is_selected = (self.cursor_row == tile_row and self.cursor_col == tile_col)
        
        # Determine tile appearance
        if is_selected and not is_blank:
            color_pair = curses.color_pair(self.COLOR_SELECTED)
            tile_text = f" {label:2d} " if label < 100 else f"{label:3d} "
        elif is_blank:
            color_pair = curses.color_pair(self.COLOR_BLANK)
            tile_text = "    "
        else:
            color_pair = curses.color_pair(self.COLOR_NORMAL)
            tile_text = f" {label:2d} " if label < 100 else f"{label:3d} "
            
        # Draw the tile
        self.stdscr.addstr(row, col, tile_text, color_pair)

    def draw_game_board(self):
        """Draw the entire game board centered in the terminal."""
        # Calculate center position
        terminal_height, terminal_width = self.stdscr.getmaxyx()
        
        tile_width = 4
        board_width = self.game.breadth * tile_width + 2  # +2 for borders
        board_height = self.game.breadth + 2  # +2 for borders
        
        # Center the board
        start_row = max(1, (terminal_height - board_height) // 2)
        start_col = max(1, (terminal_width - board_width) // 2)
        
        # Store for use in status display
        self.board_start_row = start_row
        self.board_end_row = start_row + board_height
        
        # Draw top border
        border_color = curses.color_pair(self.COLOR_BORDER)
        border_line = "+" + "-" * (board_width - 2) + "+"
        self.stdscr.addstr(start_row, start_col, border_line, border_color)
        
        # Draw game tiles and side borders
        for tile_row in range(self.game.breadth):
            screen_row = start_row + 1 + tile_row
            self.stdscr.addstr(screen_row, start_col, "|", border_color)
            
            for tile_col in range(self.game.breadth):
                screen_col = start_col + 1 + tile_col * tile_width
                self.draw_tile(screen_row, screen_col, tile_row, tile_col)
            
            self.stdscr.addstr(screen_row, start_col + board_width - 1, "|", border_color)
        
        # Draw bottom border
        bottom_row = start_row + board_height - 1
        self.stdscr.addstr(bottom_row, start_col, border_line, border_color)

    def draw_status(self):
        """Draw status information and instructions."""
        terminal_height, terminal_width = self.stdscr.getmaxyx()
        status_color = curses.color_pair(self.COLOR_STATUS)
        
        # Game status at top
        title = f"Fifteen Puzzle - Size: {self.game.breadth}x{self.game.breadth}"
        moves_text = f"Moves: {self.move_count}"
        
        # Center the title
        title_col = max(0, (terminal_width - len(title)) // 2)
        self.stdscr.addstr(0, title_col, title, status_color)
        self.stdscr.addstr(1, title_col, moves_text, status_color)
        
        # Instructions below the board
        status_start_row = getattr(self, 'board_end_row', terminal_height // 2 + 5) + 1
        
        instructions = [
            "Controls:",
            "  Arrow Keys or WASD - Move cursor",
            "  Space - Move selected tile(s)",
            "  R - Restart/Reshuffle",
            "  ESC - Return to console mode",
            "  Q - Quit completely",
            "  Ctrl+C - Exit completely"
        ]
        
        # Center the instructions
        max_instruction_len = max(len(instruction) for instruction in instructions)
        instruction_col = max(0, (terminal_width - max_instruction_len) // 2)
        
        for i, instruction in enumerate(instructions):
            if status_start_row + i < terminal_height - 1:
                self.stdscr.addstr(status_start_row + i, instruction_col, instruction, status_color)
        
        # Current selection info
        if (self.cursor_row < self.game.breadth and self.cursor_col < self.game.breadth and 
            status_start_row + len(instructions) + 2 < terminal_height - 1):
            
            selected_label = self.game.get_label(self.cursor_row, self.cursor_col)
            if selected_label != self.game.blank_label:
                move_type = self.get_move_type(selected_label)
                
                selection_text = f"Selected: Tile {selected_label}"
                move_type_text = f"Move type: {move_type}"
                
                # Center the selection info
                selection_col = max(0, (terminal_width - len(selection_text)) // 2)
                move_type_col = max(0, (terminal_width - len(move_type_text)) // 2)
                
                self.stdscr.addstr(status_start_row + len(instructions) + 1, selection_col, 
                                 selection_text, status_color)
                self.stdscr.addstr(status_start_row + len(instructions) + 2, move_type_col, 
                                 move_type_text, status_color)

    def draw_message(self, message: str, is_error: bool = False):
        """Draw a temporary message at the bottom of the screen."""
        terminal_height, terminal_width = self.stdscr.getmaxyx()
        message_row = terminal_height - 2
        color = curses.color_pair(self.COLOR_ERROR if is_error else self.COLOR_STATUS)
        
        # Clear the message line
        self.stdscr.addstr(message_row, 0, " " * min(terminal_width - 1, len(message) + 10))
        
        # Center the message
        message_col = max(0, (terminal_width - len(message)) // 2)
        if message_col + len(message) < terminal_width:
            self.stdscr.addstr(message_row, message_col, message, color)

    def can_move_tile(self, label: int) -> bool:
        """Check if a tile can be moved using the game's validation."""
        return label in self.game.get_valid_moves()

    def get_move_type(self, label: int) -> str:
        """Determine the type of move for a given tile."""
        if label == self.game.blank_label:
            return "Cannot move blank space"
            
        if not self.can_move_tile(label):
            return "Invalid move"
            
        sequence = self.game.get_move_sequence(label)
        if len(sequence) == 1:
            return "Direct swap"
        elif len(sequence) > 1:
            return f"Multi-tile shift ({len(sequence)} tiles)"
        else:
            return "Invalid move"

    def move_selected_tile(self) -> bool:
        """Move the currently selected tile if possible."""
        if self.cursor_row >= self.game.breadth or self.cursor_col >= self.game.breadth:
            return False
            
        selected_label = self.game.get_label(self.cursor_row, self.cursor_col)
        
        if selected_label == self.game.blank_label:
            self.draw_message("Cannot move the blank space!", True)
            return False
            
        if not self.can_move_tile(selected_label):
            self.draw_message("Invalid move!", True)
            return False
            
        # Use the enhanced player_move method
        success = self.game.player_move(selected_label)
        if success:
            self.move_count += 1
            sequence = self.game.get_move_sequence(selected_label)
            if len(sequence) == 1:
                self.draw_message(f"Moved tile {selected_label}")
            else:
                self.draw_message(f"Shifted {len(sequence)} tiles")
        else:
            self.draw_message("Move failed!", True)
            
        return success

    def handle_input(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        # Clear any previous message
        self.draw_message("")
        
        # Movement keys
        if key in [curses.KEY_UP, ord('w'), ord('W')]:
            self.cursor_row = max(0, self.cursor_row - 1)
        elif key in [curses.KEY_DOWN, ord('s'), ord('S')]:
            self.cursor_row = min(self.game.breadth - 1, self.cursor_row + 1)
        elif key in [curses.KEY_LEFT, ord('a'), ord('A')]:
            self.cursor_col = max(0, self.cursor_col - 1)
        elif key in [curses.KEY_RIGHT, ord('d'), ord('D')]:
            self.cursor_col = min(self.game.breadth - 1, self.cursor_col + 1)
        
        # Action keys
        elif key == ord(' '):  # Space - move tile
            self.move_selected_tile()
            
        elif key in [ord('r'), ord('R')]:  # Restart
            self.game.shuffle(self.game.shuffle_steps)
            self.move_count = 0
            self.draw_message("Board reshuffled!")
            
        elif key == 27:  # ESC - Return to console mode
            return False

        elif key in [ord('q'), ord('Q')]:  # Quit completely
            self.quit_completely = True
            return False
            
        # Check for win condition
        if self.game.is_solved():
            self.draw_message("*** CONGRATULATIONS! PUZZLE SOLVED! *** Press R to play again.", False)
            
        return True

    def run(self):
        """Main game loop for ncurses interface."""
        try:
            # Initialize ncurses
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            curses.curs_set(0)  # Hide cursor
            
            # Check minimum terminal size
            terminal_height, terminal_width = self.stdscr.getmaxyx()
            min_width = self.game.breadth * 4 + 10  # Game width + padding
            min_height = self.game.breadth + 15     # Game height + status area
            
            if terminal_width < min_width or terminal_height < min_height:
                raise Exception(f"Terminal too small. Need at least {min_width}x{min_height}, got {terminal_width}x{terminal_height}")
            
            self.setup_colors()
            
            # Main game loop
            while True:
                # Clear screen and redraw everything
                self.stdscr.clear()
                self.draw_game_board()
                self.draw_status()
                self.stdscr.refresh()
                
                # Get user input
                key = self.stdscr.getch()
                
                # Handle input (returns False to quit)
                if not self.handle_input(key):
                    break
                    
        except KeyboardInterrupt:
            pass  # Allow Ctrl+C to exit gracefully
        finally:
            # Cleanup ncurses
            if self.stdscr:
                curses.curs_set(1)  # Restore cursor
                curses.nocbreak()
                self.stdscr.keypad(False)
                curses.echo()
                curses.endwin()


def create_controller(game: Game, use_ncurses: bool = True) -> Controller:
    """Factory function to create appropriate controller."""
    if use_ncurses:
        try:
            import curses
            return NCursesController(game)
        except ImportError:
            print("NCurses not available, falling back to console mode...")
            return Controller(game)
    else:
        return Controller(game)


if __name__ == '__main__':
    # Get game size using the console controller's input method
    game_size, seed = Controller.input_game_size()
    game = Game(game_size, True, seed)
    
    # Ask user for interface preference
    print("Choose interface:")
    print("1. NCurses (visual)")
    print("2. Console (text)")
    choice = input("Enter choice [1]: ").strip()
    
    if choice == "2":
        controller = Controller(game)
        controller.play()
    else:
        try:
            controller = NCursesController(game)
            controller.run()

            # Check if user wanted to quit completely
            if controller.quit_completely:
                print("\nThank you for playing 'fifteen'. Have a nice day!")
            else:
                print("\nReturning to console mode...")
                # Continue in console mode after ncurses
                console_controller = Controller(controller.game)
                console_controller.play()
        except Exception as e:
            print(f"NCurses mode failed: {e}")
            print("Falling back to console mode...")
            controller = Controller(game)
            controller.play()
            
