import os, pygame
from Window_Constants_Definition import *
from Neural_Network_Trainer import Train_And_Save, Load_And_Eval

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
text_title = FONT_JETBRAINS_TITLE.render('SignSens', True, COLOR_TITLE)

# General Button Dimensions
button_width = 200
button_height = 50

# Button Dimensions - Start Program
button_position_x1 = screen_width * (2.5 / 3.5) - button_width / 2
button_position_y1 = screen_height * (6 / 8) - button_height / 2
button_text1 = FONT_JETBRAINS_TEXT.render('Train Model', True, COLOR_TITLE)

# Button Dimensions - Start Sign Detection
button_position_x2 = screen_width * (1 / 3.5) - button_width / 2
button_position_y2 = screen_height * (6 / 8) - button_height / 2
button_text2 = FONT_JETBRAINS_TEXT.render('Translation', True, COLOR_TITLE)

# Clearing Console
os.system('cls' if os.name == 'nt' else 'clear')

# Program Polling Loop
while not exit:        
    screen.fill(COLOR_BACKGROUND) 

    for event in pygame.event.get():
        # Getting the Mouse Positions
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Quitting program after closing window
        if event.type == pygame.QUIT:
            exit = True
            break

        # Checking for Mouse Button Down
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_position_x1 < mouse_x < button_position_x1 + button_width and button_position_y1 < mouse_y < button_position_y1 + button_height:
                # Trainer which trains the models and saves a file
                screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.HIDDEN)
                Train_And_Save()
                screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.SHOWN)
            elif button_position_x2 < mouse_x < button_position_x2 + button_width and button_position_y2 < mouse_y < button_position_y2 + button_height:
                # Loading and Evaluating the Model
                screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.HIDDEN)
                Load_And_Eval()
                screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.SHOWN)

    # Printing the Title Text
    screen.blit(text_title, (screen_width / 2 - text_title.get_size()[0] / 2, (screen_height / 5)))

    # Start Program Button
    color_button_1 = COLOR_BUTTON_HOVER if button_position_x1 < mouse_x < button_position_x1 + button_width and button_position_y1 < mouse_y < button_position_y1 + button_height else COLOR_BUTTON_IDLE
    pygame.draw.rect(screen, color_button_1, (button_position_x1, button_position_y1, button_width, button_height))
    screen.blit(button_text1, 
                (button_position_x1 + (button_width - button_text1.get_width()) / 2, button_position_y1 + (button_height - button_text1.get_height()) / 2))

    # Sign Detection Button
    color_button_2 = COLOR_BUTTON_HOVER if button_position_x2 < mouse_x < button_position_x2 + button_width and button_position_y2 < mouse_y < button_position_y2 + button_height else COLOR_BUTTON_IDLE
    pygame.draw.rect(screen, color_button_2, (button_position_x2, button_position_y2, button_width, button_height))
    screen.blit(button_text2, 
                (button_position_x2 + (button_width - button_text2.get_width()) / 2, button_position_y2 + (button_height - button_text2.get_height()) / 2))

    # Updating the screen each frame
    pygame.display.update()

pygame.quit()

# Clearing Console
os.system('cls' if os.name == 'nt' else 'clear')