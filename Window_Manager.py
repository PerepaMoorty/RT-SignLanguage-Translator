import pygame
from Window_Constants_Definition import *

def main():
    # Initializing Pygame
    pygame.init()

    # Defining the resolution of the screen
    screen_width = 640
    screen_height = 360
    
    # Creating a canvas
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("SignSens")
    exit = False

    # Defining texts
    text_title = FONT_JETBRAINS.render('SignSens', True, COLOR_TITLE)

    # Program Polling Loop
    while not exit:        
        screen.fill(COLOR_BACKGROUND) 
        
        for event in pygame.event.get():
            # Quiting program after closing window
            if event.type == pygame.QUIT:
                exit = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Getting the Mouse Positions
                mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Printing the Title Text
        screen.blit(text_title, (screen_width / 2 - text_title.get_size()[0] / 2, (screen_height / 8)))
        
        # Start Training
        
        # Updating the screen each frame
        pygame.display.update()
        

main()