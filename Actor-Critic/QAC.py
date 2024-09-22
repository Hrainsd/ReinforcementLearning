import numpy as np
import random
import matplotlib.pyplot as plt

# Define the gridworld
gridworld = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, -1, -1, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, -1, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

# Define parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
alpha_w = 0.1  # Learning rate for critic (value update)
alpha_theta = 0.01  # Learning rate for actor (policy update)
episodes = 10000
max_steps = 100

# Define action space
actions = ['Up', 'Down', 'Left', 'Right']
action_effects = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Left': (0, -1),
    'Right': (0, 1)
}

# Initialize Q-values (critic) and policy (actor)
w = np.zeros((gridworld.shape[0], gridworld.shape[1], len(actions)))
theta = np.random.uniform(low=0, high=0.1, size=(gridworld.shape[0], gridworld.shape[1], len(actions)))

# Function to choose action based on softmax policy
def choose_action(state):
    theta_state = theta[state[0], state[1], :]
    max_theta = np.max(theta_state)
    exp_theta = np.exp(theta_state - max_theta)
    probs = exp_theta / np.sum(exp_theta)
    action_index = np.random.choice(len(actions), p=probs)
    return actions[action_index]

# QAC learning algorithm
for episode in range(episodes):
    # Randomly start from any non-terminal state
    initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                      gridworld[i, j] == 0]
    state = random.choice(initial_states)

    for step in range(max_steps):
        action = choose_action(state)
        action_index = actions.index(action)

        # Take action and observe next state and reward
        next_state = (max(0, min(state[0] + action_effects[action][0], gridworld.shape[0] - 1)),
                     max(0, min(state[1] + action_effects[action][1], gridworld.shape[1] - 1)))
        reward = gridworld[next_state[0], next_state[1]]

        # Critic update (Q-value update)
        delta = reward + gamma * np.max(w[next_state[0], next_state[1], :]) - w[state[0], state[1], action_index]
        w[state[0], state[1], action_index] += alpha_w * delta

        # Actor update (policy update)
        theta[state[0], state[1], :] += alpha_theta * w[state[0], state[1], :]

        # Normalize policy (optional but recommended)
        # theta[state[0], state[1], :] = np.maximum(theta[state[0], state[1], :], 1e-10)
        # theta[state[0], state[1], :] /= np.sum(theta[state[0], state[1], :])

        state = next_state

        if reward == rtarget:
            break

# Extract the optimal policy and compute state values
policy_grid = np.full(gridworld.shape, '', dtype=object)
value_grid = np.zeros(gridworld.shape)

for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            state = (i, j)
            # Compute state value by weighted average of Q-values
            theta_state = theta[i, j, :]
            probs = np.exp(theta_state - np.max(theta_state)) / np.sum(np.exp(theta_state - np.max(theta_state)))
            value_grid[i, j] = np.sum(w[i, j, :] * probs)
            # Extract policy
            best_action_index = np.argmax(w[i, j])
            policy_grid[i, j] = actions[best_action_index]

# Print the Q-values
print("Critic Values (w):")
print(w)

# Print the optimal policy
print("\nOptimal Policy:")
print(policy_grid)

# Print the state values
print("\nState Values:")
print(value_grid)

# Plot state value and policy
fig, ax = plt.subplots()
cmap = plt.cm.winter
norm = plt.Normalize(vmin=value_grid.min(), vmax=value_grid.max())
cbar = ax.imshow(value_grid, cmap=cmap, norm=norm, interpolation='nearest')

# Add the policy arrows
for i in range(policy_grid.shape[0]):
    for j in range(policy_grid.shape[1]):
        if policy_grid[i, j] in actions:
            if policy_grid[i, j] == 'Up':
                ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'Down':
                ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'Left':
                ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'Right':
                ax.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')

# Display grid and labels
ax.set_xticks(np.arange(gridworld.shape[1]))
ax.set_yticks(np.arange(gridworld.shape[0]))
ax.set_xticklabels(np.arange(1, gridworld.shape[1] + 1))
ax.set_yticklabels(np.arange(1, gridworld.shape[0] + 1))
ax.grid(which='both', color='black', linestyle='-', linewidth=2)

# Set terminal state labels and colors
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == -1:
            ax.text(j, i, 'T', va='center', ha='center', color='white')
        elif gridworld[i, j] == 1:
            ax.text(j, i, 'G', va='center', ha='center', color='white')

plt.title('Policy Visualization')
plt.colorbar(cbar, ax=ax, label='State Value')
plt.show()

# Plot the state value
plt.imshow(value_grid, cmap='winter', interpolation='none')
plt.colorbar(label='State Value')

# Annotate each grid cell with the value
for x in range(value_grid.shape[0]):
    for y in range(value_grid.shape[1]):
        plt.text(y, x, f'{value_grid[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('State Value')
plt.show()
