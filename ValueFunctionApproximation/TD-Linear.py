import numpy as np
import random

# Gridworld definition
gridworld = np.array([
    [0, 0, 0],
    [0, 0, -1],
    [-1, 0, 1]
])

# Parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
alpha = 0.1  # Learning rate
epsilon = 0.1  # Epsilon-greedy parameter
episodes = 10000

# State-space and action-space
states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if gridworld[i, j] == 0]
actions = ['Up', 'Down', 'Left', 'Right']
state_values = np.zeros(gridworld.shape)

def get_next_state(state, action):
    i, j = state
    if action == 'Up':
        return (max(0, i - 1), j)
    elif action == 'Down':
        return (min(gridworld.shape[0] - 1, i + 1), j)
    elif action == 'Left':
        return (i, max(0, j - 1))
    elif action == 'Right':
        return (i, min(gridworld.shape[1] - 1, j + 1))

def get_reward(state, next_state):
    if state == next_state:
        return rboundary
    elif gridworld[next_state] == rforbidden:
        return rforbidden
    elif gridworld[next_state] == rtarget:
        return rtarget
    else:
        return 0

def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        values = []
        for action in actions:
            next_state = get_next_state(state, action)
            values.append(state_values[next_state])
        return actions[np.argmax(values)]

def td_linear(episodes):
    for _ in range(episodes):  # Number of episodes
        # state = (0,0)
        # Choose a random initial state that is not a terminal or forbidden state
        initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                          gridworld[i, j] == 0]
        state = random.choice(initial_states)
        while True:
            action = epsilon_greedy(state)
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            # Update the state value based on TD learning rule
            state_values[state] += alpha * (reward + gamma * state_values[next_state] - state_values[state])
            if gridworld[next_state] == rtarget or gridworld[next_state] == rforbidden:
                break
            state = next_state

    return state_values

# Run the TD-Linear algorithm
final_values = td_linear(episodes)
print("Final state values:")
print(np.round(final_values, 4))
