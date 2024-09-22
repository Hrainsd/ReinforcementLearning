import numpy as np
import random
import matplotlib.pyplot as plt

# Define the gridworld
gridworld = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, -1, -1, -1, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1, -1,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1]
])

# Define parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
alpha = 0.1
epsilon = 0.1
episodes = 50000
max_steps = 1000

# Define actions
actions = ['Up', 'Down', 'Left', 'Right']
action_effects = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Left': (0, -1),
    'Right': (0, 1)
}

# Initialize Q-function approximator (weights)
def initialize_weights(n_features, n_actions):
    return np.random.uniform(low=0.0, high=0.1, size=(n_actions, n_features))

def feature_vector(state):
    features = np.zeros(gridworld.shape[0] * gridworld.shape[1])
    index = state[0] * gridworld.shape[1] + state[1]
    features[index] = 1
    return features

# Initialize weights
n_features = gridworld.shape[0] * gridworld.shape[1]
weights = initialize_weights(n_features, len(actions))

# Epsilon-greedy policy
def epsilon_greedy_policy(state, weights, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        q_values = [np.dot(feature_vector(state), weights[action]) for action in range(len(actions))]
        return np.argmax(q_values)

# Get next state
def get_next_state(state, action):
    row, col = state
    drow, dcol = action_effects[actions[action]]
    new_row, new_col = row + drow, col + dcol

    # Check boundaries
    if new_row < 0 or new_row >= gridworld.shape[0] or new_col < 0 or new_col >= gridworld.shape[1]:
        return state, rboundary

    # Check for forbidden states
    if gridworld[new_row, new_col] == rforbidden:
        return state, rboundary

    return (new_row, new_col), gridworld[new_row, new_col]

# Q-Learning algorithm with function approximation
def q_learning(gridworld, weights, episodes, alpha, gamma, epsilon, max_steps):
    for episode in range(episodes):
        # Choose a random initial state that is not a terminal or forbidden state
        initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                          gridworld[i, j] == 0]
        state = random.choice(initial_states)

        for step in range(max_steps):
            action = epsilon_greedy_policy(state, weights, epsilon)
            next_state, reward = get_next_state(state, action)

            # Compute Q-values
            q_value = np.dot(feature_vector(state), weights[action])
            next_q_values = [np.dot(feature_vector(next_state), weights[a]) for a in range(len(actions))]
            next_q_value = np.max(next_q_values)

            # Update weights
            td_target = reward + gamma * next_q_value
            td_error = td_target - q_value
            weights[action] += alpha * td_error * feature_vector(state)

            state = next_state

            if reward == rtarget:
                break

    return weights

# Training the agent
weights = q_learning(gridworld, weights, episodes, alpha, gamma, epsilon, max_steps)

# Print the learned weights
print("Weights:")
for action in range(len(actions)):
    print(f"Action {actions[action]}: {weights[action]}")

# Print the Q-values
print("\nQ-values:")
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            state = (i, j)
            q_values = [np.dot(feature_vector(state), weights[a]) for a in range(len(actions))]
            print(f"State {state}: {q_values}")

# Extract the optimal policy
policy_grid = np.full(gridworld.shape, '', dtype=object)

for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            state = (i, j)
            best_action = np.argmax([np.dot(feature_vector(state), weights[a]) for a in range(len(actions))])
            policy_grid[i, j] = actions[best_action].lower()

# Print the optimal policy
print("\nOptimal Policy:")
print(policy_grid)

# Compute state values from Q-values
def compute_state_values(weights):
    state_values = np.zeros(gridworld.shape)

    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] == 0:  # Only calculate for non-terminal states
                state = (i, j)
                q_values = [np.dot(feature_vector(state), weights[action]) for action in range(len(actions))]
                state_values[i, j] = np.mean(q_values)  # Average over all actions

    return state_values

# Compute and print state values
state_values = compute_state_values(weights)
print("\nState Values:")
print(state_values)


# Plot state value and policy
fig, ax = plt.subplots()
cmap = plt.cm.Spectral
norm = plt.Normalize(vmin=state_values.min(), vmax=state_values.max())
cbar = ax.imshow(state_values, cmap=cmap, norm=norm, interpolation='nearest')

# Add the policy arrows
for i in range(policy_grid.shape[0]):
    for j in range(policy_grid.shape[1]):
        if policy_grid[i, j]:
            if policy_grid[i, j] == 'up':
                ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'down':
                ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'left':
                ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'right':
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


# Plot state values
plt.imshow(state_values, cmap='winter', interpolation='none')
plt.colorbar(label='State Value')

# Annotate each grid cell with the value
for x in range(state_values.shape[0]):
    for y in range(state_values.shape[1]):
        if gridworld[x, y] == 0:  # Only annotate non-terminal states
            plt.text(y, x, f'{state_values[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('State Value')
plt.show()
