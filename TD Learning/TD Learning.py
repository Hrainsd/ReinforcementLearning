import numpy as np
import matplotlib.pyplot as plt

# Define the gridworld
gridworld = np.array([
    [0, 0, 0],
    [0, 0, -1],
    [-1, 0, 1]
])

# Define parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
alpha = 0.1  # Learning rate
num_episodes = 1000

# Initialize state value function
V = np.zeros(gridworld.shape)

# Define possible actions
actions = ['Up', 'Down', 'Left', 'Right']

# Define the transition probabilities and rewards
def get_next_state_and_reward(state, action):
    x, y = state
    if action == 'Up':
        next_state = (x - 1, y)
    elif action == 'Down':
        next_state = (x + 1, y)
    elif action == 'Left':
        next_state = (x, y - 1)
    elif action == 'Right':
        next_state = (x, y + 1)

    # Check for boundary conditions
    if next_state[0] < 0 or next_state[0] >= gridworld.shape[0] or next_state[1] < 0 or next_state[1] >= gridworld.shape[1]:
        return state, rboundary  # Return to the same state with boundary reward

    # Check for forbidden states
    if gridworld[next_state] == -1:
        return state, rforbidden  # Return to the same state with forbidden reward

    # Check for target states
    if gridworld[next_state] == 1:
        return next_state, rtarget

    # Normal movement with no special rewards
    return next_state, 0

# Define the policy (random policy for this example)
def choose_action(state):
    return np.random.choice(actions)

# TD(0) Learning
# Iterate over each state as the starting state
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        # Skip forbidden states
        if gridworld[(i, j)] == -1 or gridworld[(i, j)] == 1:
            continue

        for episode in range(num_episodes):
            state = (i, j)
            while True:
                action = choose_action(state)
                next_state, reward = get_next_state_and_reward(state, action)

                # TD(0) update rule
                V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])

                if gridworld[next_state] == 1:  # Stop if the target state is reached
                    break

                state = next_state

        # Print the state value function after each state has been processed
        print(f"Start state: {(i, j)}\nState value：")
        print(V)
        print()

        # Plot the state value function
        plt.imshow(V, cmap='winter', interpolation='none')
        plt.colorbar(label='State Value')

        # Annotate each grid cell with the value
        for x in range(V.shape[0]):
            for y in range(V.shape[1]):
                plt.text(y, x, f'{V[x, y]:.2f}', ha='center', va='center', color='yellow')

        plt.title(f'State Value Function - After state {(i, j)}')
        plt.show()
