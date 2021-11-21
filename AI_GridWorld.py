from GridWorld import GridWorld
import numpy as np
import pygame
from collections import defaultdict

# Values
samples = 10
score_threshold = 100

gamma = 0.9
epsilon = 0.1
score = 0
iteration_array = np.array([])

# Initialize the environment
env = GridWorld()
env.reset()
x, y, has_key = env.get_state()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
done = False

# Game clock
clock = pygame.time.Clock()

iteration_max = 1000
iteration = 0
sample = 0
for i in range(samples):
    # Reset and define data structure for q-table
    q_table = defaultdict(lambda: np.zeros(len(actions))) # Create table with default value of 0 for each action
    while iteration < iteration_max:            
        # Automatic reset environment in AI mode
        if done:
            env.reset()
            x, y, has_key = env.get_state()
        
        # 1. choose an action
        q_current = q_table[(x, y, has_key)] # get current q-value for the state
        if np.random.uniform(0, 1) < epsilon: # choose random action with probability epsilon (Greedy action)
            action_index = np.random.choice(len(actions))
        else:
            action_index = np.argmax(q_current) # get index of the best action
        action = actions[action_index]
        # 2. step the environment
        (x, y, has_key), reward, done = env.step(action) # take the action
        # 3. update q table
        q_next = q_table [(x, y, has_key)] # get next q-value for the state
        q_current[action_index] = reward + gamma * np.max(q_next) # update q-value for the state

        # Update the score
        score += reward

        # Log the score and iteration
        if done:
            if score > score_threshold:
                print("Sample: ", sample + 1, "Iteration: ", iteration + 1, "Score: ", score)
                iteration_array = np.append(iteration_array, iteration + 1)
                iteration = 0
                sample += 1
                break
            # else:
            #     print("Iteration: ", iteration + 1, "Score: ", score)
            score = 0
            iteration += 1

# Print the iteration array to file
with open("AI_GridWorld_QLearning_data.txt", "w") as file:
    for i in iteration_array:
        file.write("%d\n" % i)
        
env.close()