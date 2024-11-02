import pygame

def main():
    # Initializing Pygame
    pygame.init()

    # Defining the resolution of the screen
    screen_width = 1080
    screen_height = 720
    
    # Creating a canvas
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("SignSens")
    exit = False

    # Defining Colors [RGB Format]
    # Primary Tints
    PRIMARY_A0 = (175, 82, 193)
    PRIMARY_A20 = (185, 102, 200)
    PRIMARY_A40 = (195, 122, 207)
    PRIMARY_A60 = (204, 141, 214)
    PRIMARY_A80 = (213, 160, 221)
    PRIMARY_A100 = (222, 179, 228)

    # Surface Colors
    SURFACE_A0 = (18, 18, 18)
    SURFACE_A20 = (40, 40, 40)
    SURFACE_A40 = (63, 63, 63)
    SURFACE_A60 = (87, 87, 87)
    SURFACE_A80 = (113, 113, 113)
    SURFACE_A100 = (139, 139, 139)

    # Mixed Surface Colors
    MIXED_A0 = (32, 25, 33)
    MIXED_A20 = (53, 46, 54)
    MIXED_A40 = (75, 69, 76)
    MIXED_A60 = (98, 92, 99)
    MIXED_A80 = (122, 117, 123)
    MIXED_A100 = (147, 143, 148)

    # Defining Fonts
    FONT_JETBRAINS = pygame.font.SysFont('Jetbrains Mono', 35)
    
    # Defining texts
    text_title = FONT_JETBRAINS.render('SignSens', True, PRIMARY_A80)

    # Program Polling Loop
    while not exit:        
        screen.fill(SURFACE_A0) 
        
        for event in pygame.event.get():
            # Quiting program after closing window
            if event.type == pygame.QUIT:
                exit = True
        
        # Printing the Title Text
        screen.blit(text_title, (screen_width / 2 - text_title.get_size()[0] / 2, (screen_height / 32)))
        
        
        
        # Updating the screen each frame
        pygame.display.update()
        
        
main()