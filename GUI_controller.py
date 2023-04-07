# GUI_controller.py 

from Game import Game
import pygame
import time

def change_volume(volume):
    pygame.mixer.music.set_volume(volume)
    click_sound.set_volume(volume)
    win_sound.set_volume(volume)

def draw_grid():
    gap = (width - 160) // rows
    for vertical_step in range(rows):
        pygame.draw.line(window, grey, (0, vertical_step * gap), (width, vertical_step * gap), margin)
        for horizontal_step in range(rows):
            pygame.draw.line(window, grey, (horizontal_step * gap, 0), (horizontal_step * gap, width), margin)
    pygame.draw.rect(window, grey, (480, 0, 160, 480))


def draw_tiles():
    text_margin = (tile_breadth // 2)
    start = [text_margin, text_margin]
    for local_x in range(game.breadth):
        for local_y in range(game.breadth):
            label = game.get_label(local_x, local_y)
            img = font.render(str(label), True, black)
            if label is not game.blank_label:
                window.blit(img, (start[0], start[1]))
            start[0] += tile_breadth + margin
        start[0] -= (tile_breadth + margin) * game.breadth
        start[1] += tile_breadth + margin


def frame_update():
    global won, run, volume
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    click = pygame.mouse.get_pressed(3)
    mouse_x, mouse_y = pygame.mouse.get_pos()
    
    # # Volume control
    # if 520 <= mouse_x <= 560 and 20 <= mouse_y <= 60:
    #     volume = volume - 0.1 if volume > 0 else 1.0
    #     change_volume(volume)
    #     print(f"Volume set to {volume * 100:.0f}%")

    if click == (1, 0, 0):
        clock.tick(fps)
        time.sleep(0.05)
        tile_pos_x = mouse_x // tile_breadth
        tile_pos_y = mouse_y // tile_breadth
        tile = game.get_label(tile_pos_y, tile_pos_x)
        if tile in game.get_valid_moves():
            pygame.mixer.Sound.play(click_sound)
            print(f"Mouse-click: row, column ({tile_pos_x}, {tile_pos_y}). Target tile label: {tile}")

        if tile_pos_y == 0 and tile_pos_x == 0 and game.is_solved():
            game.shuffle(game.shuffle_default)
            won = False
        else:
            game.slide_tile(tile)
        if game.is_solved() and not won:
            pygame.mixer.Sound.play(win_sound)
            won = True

def play_game():
    global run
    run = True
    while run:
        clock.tick(fps)
        frame_update()
        window.fill(cyan)
        draw_grid()
        draw_tiles()
        # draw_volume_button()
        pygame.display.update()

def draw_volume_button():
    volume_icon = pygame.image.load('images/volume_icon.png').convert_alpha()
    volume_icon = pygame.transform.scale(volume_icon, (40, 40))
    window.blit(volume_icon, (width - 80, 20))


if __name__ == '__main__':
    black   = (0, 0, 0)
    white   = (255, 255, 255)
    grey    = (100, 100, 100)
    red     = (200, 0, 0)
    green   = (0, 200, 0)
    yellow  = (200, 200, 0)
    blue    = (0, 0, 200)
    cyan    = (0, 200, 200)
    magenta = (200, 0, 200)
    purple  = (200, 0, 100)
    margin  = 3
    # TODO add a GUI feature for selecting a different sized game. Suggested: add dynamic (moving) tiles, first.
    this_breadth = 4  # A breadth of 4 is the default matrix for Fifteen.
    window_breadth = (640, 480)
    width = window_breadth[0]
    height = window_breadth[1]
    fps = 30
    game = Game(this_breadth, True)
    tile_breadth = (width - 160 - (margin * 2 * game.breadth)) // game.breadth
    rows = game.breadth
    run = True
    won = False
    volume = 0.12
    pygame.init()
    pygame.mixer.init()
    click_sound = pygame.mixer.Sound("audio/click.wav")
    win_sound = pygame.mixer.Sound("audio/tada.wav")
    change_volume(volume)
    window = pygame.display.set_mode(window_breadth)
    pygame.display.set_caption("Python 15 Puzzle")
    icon = pygame.image.load('images/launch.png').convert()  # https://www.flaticon.com/authors/icongeek26
    pygame.display.set_icon(icon)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Courier", 36)
    play_game()
