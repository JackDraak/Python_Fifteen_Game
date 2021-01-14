import main
import pygame
import time


def draw_grid():
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(screen, grey, (0, i * gap), (width, i * gap), margin)
        for j in range(rows):
            pygame.draw.line(screen, grey, (j * gap, 0), (j * gap, width), margin)


def draw_tiles():
    global label
    text_margin = (tile_dimension // 2) + (margin * 2)
    start = [text_margin, text_margin]
    for x in range(g.dimension):
        for y in range(g.dimension):
            label = g.get_label(x, y)
            img = font.render(str(label), True, black)
            if label is not g.blank_label:
                screen.blit(img, (start[0], start[1]))
            start[0] += tile_dimension + margin
        start[0] -= (tile_dimension + margin) * g.dimension
        start[1] += tile_dimension + margin


def frame_update():
    global run, label
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    click = pygame.mouse.get_pressed()
    if click == (1, 0, 0):
        pygame.mixer.Sound.play(click_sound)
        clock.tick(fps)
        time.sleep(0.12)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        tile_pos_x = mouse_x // tile_dimension
        tile_pos_y = mouse_y // tile_dimension
        label = g.get_label(tile_pos_y, tile_pos_x)
        print(f"Mouse-click: row, column ({tile_pos_x}, {tile_pos_y}). Target tile label: {label}")
        if tile_pos_y == 0 and tile_pos_x == 0 and g.is_solved():
            g.shuffle(g.shuffle_default)
        g.slide_tile(label)
        if g.is_solved():
            pygame.mixer.Sound.play(win_sound)


if __name__ == '__main__':
    width = 600
    height = 600
    fps = 30

    black = (0, 0, 0)
    white = (255, 255, 255)
    grey = (128, 128, 128)
    red = (200, 0, 0)
    green = (0, 200, 0)
    yellow = (200, 200, 0)
    blue = (0, 0, 200)
    cyan = (0, 200, 200)
    magenta = (200, 0, 200)
    margin = 5

    # TODO add a GUI feature for selecting a different sized game.
    g = main.Game(3, True)  # A dimension of 3 is the default matrix for Fifteen.
    tile_dimension = (width - (margin * 2 * g.dimension)) // g.dimension
    rows = g.dimension
    label = 0

    pygame.init()
    pygame.mixer.init()
    click_sound = pygame.mixer.Sound("audio/click.wav")
    win_sound = pygame.mixer.Sound("audio/tada.wav")
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Python 15 Puzzle")
    icon = pygame.image.load('images/launch.png').convert()  # https://www.flaticon.com/authors/icongeek26
    pygame.display.set_icon(icon)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 60)

    run = True
    while run:
        clock.tick(fps)
        frame_update()
        screen.fill(cyan)
        draw_grid()
        draw_tiles()
        pygame.display.update()
