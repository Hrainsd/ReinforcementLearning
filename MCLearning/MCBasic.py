import numpy as np
import matplotlib.pyplot as plt

# Gridworld definition
gridworld = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, -1, -1, 0],
    [0, 0, -1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1]
])

# Define parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
episodes = 1000
max_steps = 100  # Maximum steps per episode

# Define actions and their corresponding names in the order: Up, Down, Left, Right
action_tuples = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
action_names = ["Up", "Down", "Left", "Right"]
action_map = dict(zip(action_tuples, action_names))

# Initialize policy randomly for non-forbidden and non-target states
policy = {}
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] != rforbidden and gridworld[i, j] != rtarget:
            policy[(i, j)] = tuple(action_tuples[np.random.choice(len(action_tuples))])

# Initialize Q-value function arbitrarily for non-forbidden states
Q = {}
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] != rforbidden and gridworld[i, j] != rtarget:
            for action in action_tuples:
                Q[((i, j), action)] = np.random.rand()  # Random initialization

# Define function to generate an episode from a specific state-action pair
def generate_episode_from_state_action(state, action, policy):
    episode = []
    current_state = state
    current_action = action
    while gridworld[current_state] != rtarget:  # Until reaching the target state
        next_state = (current_state[0] + current_action[0], current_state[1] + current_action[1])
        # Check boundaries and forbidden areas
        if (next_state[0] < 0 or next_state[0] >= gridworld.shape[0] or
                next_state[1] < 0 or next_state[1] >= gridworld.shape[1] or
                gridworld[next_state] == rforbidden):
            reward = rboundary
            next_state = current_state  # Stay in the current state
        else:
            reward = gridworld[next_state]

        episode.append((current_state, current_action, reward))
        current_state = next_state
        if gridworld[current_state] != rtarget:
            current_action = policy[current_state]  # Update action based on policy

        # Prevent excessive episode length
        if len(episode) > max_steps:  # Arbitrary large number to prevent infinite episodes
            break

    return episode

# Monte Carlo Basic Algorithm
for k in range(episodes):  # Run for a fixed number of iterations
    print(f"Iteration: {k+1}") # Add this line to monitor the iterations
    # Policy evaluation
    for state in policy.keys():
        for action in action_tuples:
            action = tuple(action)  # Ensure action is a tuple
            # Collect episodes starting from (s, a) following pi_k
            returns = []
            for _ in range(max_steps):  # Generate a fixed number of episodes
                episode = generate_episode_from_state_action(state, action, policy)
                G = 0
                for t in range(len(episode) - 1, -1, -1):
                    G = gamma * G + episode[t][2]
                    if episode[t][0] == state and episode[t][1] == action:
                        returns.append(G)
                        break # Break after finding the first (state, action) occurrence
            # Calculate average return
            if returns:  # Only update if returns is not empty
                Q[(state, action)] = np.mean(returns)

    # Policy improvement
    policy_stable = True
    for state in policy.keys():
        # Find the action with the highest Q-value
        old_action = policy[state]
        best_action = max(action_tuples, key=lambda a: Q[(state, tuple(a))])
        if old_action != best_action:
            policy_stable = False
        policy[state] = tuple(best_action)

    if policy_stable:
        print("Policy converged after {} iterations.".format(k+1))
        break

# Print the state-action values
print("State-Action Values:")
for state in policy.keys():
    state_action_values = {action_map[action]: Q[(state, action)] for action in action_tuples}
    print(f"State {state}: {state_action_values}")

# Create the grid for final policy
policy_grid = np.full(gridworld.shape, '', dtype=object)
for state, action in policy.items():
    policy_grid[state] = action_map[action]

# Print the final policy
print("\nFinal Policy:")
print(policy_grid)

# Compute state values from Q-values using the policy
def compute_state_values(Q, policy):
    state_values = np.zeros(gridworld.shape)

    for state in policy.keys():
        action = policy[state]
        state_values[state] = Q[(state, action)]  # State value is the Q-value of the action specified by the policy

    return state_values

# Compute and print state values
state_values = compute_state_values(Q, policy)
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

# Plot state values
plt.imshow(state_values, cmap='winter', interpolation='none')
plt.colorbar(label='State Value')

# Annotate each grid cell with the value
for x in range(state_values.shape[0]):
    for y in range(state_values.shape[1]):
        if gridworld[x, y] != rforbidden and gridworld[x, y] != rtarget:  # Annotate only non-forbidden and non-target states
            plt.text(y, x, f'{state_values[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('State Value')
plt.show()
