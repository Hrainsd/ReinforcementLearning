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
alpha_v = 0.1  # Learning rate for critic (value update)
alpha_theta = 0.01  # Learning rate for actor (policy update)
episodes = 30000
max_steps = 100

# Define action space
actions = ['Up', 'Down', 'Left', 'Right']
action_effects = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Left': (0, -1),
    'Right': (0, 1)
}

# Initialize state values (critic) and policy (actor)
v = np.zeros((gridworld.shape[0], gridworld.shape[1]))
theta = np.random.uniform(low=0, high=0.1, size=(gridworld.shape[0], gridworld.shape[1], len(actions)))

# Function to choose action based on softmax policy
def choose_action(state):
    theta_state = theta[state[0], state[1], :]
    max_theta = np.max(theta_state)
    exp_theta = np.exp(theta_state - max_theta)
    probs = exp_theta / np.sum(exp_theta)
    action_index = np.random.choice(len(actions), p=probs)
    return actions[action_index]

# Function to compute the advantage function
def compute_advantage(reward, v_current, v_next):
    return reward + gamma * v_next - v_current

# A2C learning algorithm
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

        # Critic update (state value update)
        advantage = compute_advantage(reward, v[state[0], state[1]], v[next_state[0], next_state[1]])
        v[state[0], state[1]] += alpha_v * advantage

        # Actor update (policy update)
        theta[state[0], state[1], action_index] += alpha_theta * advantage

        # Normalize policy
        theta_sum = np.sum(theta[state[0], state[1], :])
        if theta_sum > 0:
            theta[state[0], state[1], :] /= theta_sum

        state = next_state

        if reward == rtarget:
            break

# Compute Q-values
Q_values = np.zeros((gridworld.shape[0], gridworld.shape[1], len(actions)))
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            for action_index in range(len(actions)):
                action = actions[action_index]
                next_state = (max(0, min(i + action_effects[action][0], gridworld.shape[0] - 1)),
                             max(0, min(j + action_effects[action][1], gridworld.shape[1] - 1)))
                reward = gridworld[next_state[0], next_state[1]]
                advantage = compute_advantage(reward, v[i, j], v[next_state[0], next_state[1]])
                Q_values[i, j, action_index] = v[i, j] + advantage

# Print Q-values
print("Q-values:")
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            print(f"State ({i}, {j}): {Q_values[i, j, :]}")

# Print the optimal policy
policy_grid = np.full(gridworld.shape, '', dtype=object)
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            state = (i, j)
            best_action_index = np.argmax(theta[i, j])
            policy_grid[i, j] = actions[best_action_index]

print("\nOptimal Policy:")
print(policy_grid)

# Print the state values
print("\nState Values:")
print(v)

# Plot state value and policy
fig, ax = plt.subplots()
cmap = plt.cm.winter
norm = plt.Normalize(vmin=v.min(), vmax=v.max())
cbar = ax.imshow(v, cmap=cmap, norm=norm, interpolation='nearest')

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
plt.imshow(v, cmap='winter', interpolation='none')
plt.colorbar(label='State Value')

# Annotate each grid cell with the value
for x in range(v.shape[0]):
    for y in range(v.shape[1]):
        plt.text(y, x, f'{v[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('State Value')
plt.show()
