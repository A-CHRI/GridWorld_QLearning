# Grid World: Manual play

# Instructions:
#   Move up, down, left, or right to move the character. The 
#   objective is to find the key and get to the door
#
# Control:
#    arrows  : Merge up, down, left, or right
#    r       : Restart game
#    q / ESC : Quit

from GridWorld import GridWorld
import pygame

env = GridWorld()
env.reset()
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
while not exit_program:
    env.render()

    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_UP:
                action, action_taken = 'up', True
            if event.key == pygame.K_DOWN:
                action, action_taken  = 'down', True
            if event.key == pygame.K_RIGHT:
                action, action_taken  = 'right', True
            if event.key == pygame.K_LEFT:
                action, action_taken  = 'left', True
            if event.key == pygame.K_r:
                env.reset()   
                
    if action_taken:
        (x, y, has_key), reward, done = env.step(action)
        action_taken = False
    
env.close()
