# Grid World: AI-controlled play

# Instructions:
#   Move up, down, left, or right to move the character. The 
#   objective is to find the key and get to the door
#
# Control:
#    a       : Toggle AI player
#    d       : Toggle rendering 
#    r       : Restart game
#    q / ESC : Quit

from GridWorld import GridWorld
import numpy as np
import pygame
from collections import defaultdict

# Values
gamma = 0.9
score = 0
score_array = np.array([])

# Initialize the environment
env = GridWorld()
env.reset()
x, y, has_key = env.get_state()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
runai = True
render = False
done = False

# Game clock
clock = pygame.time.Clock()

# Define data structure for q-table
q_table = defaultdict(lambda: np.zeros(len(actions))) # Create table with default value of 0 for each action

iteration_max = 200
iteration = 0
while iteration < iteration_max:
    if render:
        env.render()
        
    # Automatic reset environment in AI mode
    if done and runai:
        env.reset()
        x, y, has_key = env.get_state()
        
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_r:
                env.reset()
    
    # AI controller (enable/disable by pressing 'a')
    if runai:
        # 1. choose an action
        q_current = q_table[(x, y, has_key)] # get current q-value for the state
        action_index = np.argmax(q_current) # get index of the best action
        action = actions[action_index]
        # 2. step the environment
        (x, y, has_key), reward, done = env.step(action) # take the action
        # 3. update q table
        q_next = q_table[(x, y, has_key)] # get next q-value for the state
        q_current[action_index] = reward + gamma * np.max(q_next) # update q-value for the state

        score += reward

        if done == True:
            print("Iteration: ", iteration + 1)
            score_array = np.append(score_array, score)
            score = 0
            iteration += 1
        
print(score_array)

env.close()
