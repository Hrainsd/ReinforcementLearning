import numpy as np
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

gamma = 0.9
epsilon = 1e-6

# Initialize value function and policy
V = np.zeros(gridworld.shape)
policy = np.full(gridworld.shape, '', dtype=object)

# Initialize Q-values dictionary
Q = {}
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        Q[(i, j)] = np.zeros(4)  # 4 actions: right, left, down, up

# Lists to store values and returns
value_iterations = []
returns = []

# Value iteration with policy extraction
iterations = 0
while True:
    delta = 0
    new_policy = np.copy(policy)
    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] in [-1, 1]:  # Skip terminal states
                continue

            max_q_value = float("-inf")
            best_action = None
            for idx, (action, action_str) in enumerate(zip([(0, 1), (0, -1), (1, 0), (-1, 0)], ['right', 'left', 'down', 'up'])):
                ni, nj = i + action[0], j + action[1]
                if 0 <= ni < gridworld.shape[0] and 0 <= nj < gridworld.shape[1]:
                    if gridworld[ni, nj] != -1:
                        q_value = gridworld[ni, nj] + gamma * V[ni, nj]
                        Q[(i, j)][idx] = q_value  # Store Q-value
                        if q_value > max_q_value:
                            max_q_value = q_value
                            best_action = action_str

            new_value = max_q_value
            delta = max(delta, abs(new_value - V[i, j]))
            V[i, j] = new_value
            new_policy[i, j] = best_action

    value_iterations.append(np.copy(V))
    returns.append(V.sum())
    iterations += 1
    policy = new_policy
    if delta < epsilon:
        break

# Print Q-values
print("\nQ-values:")
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:  # Only print for non-terminal states
            print(f"State ({i}, {j}): {Q[(i, j)]}")

print("Optimal policy:")
print(policy)

print("State value:")
print(V)

print(f"Total iterations: {iterations}")


# Plot state value and policy
fig, ax = plt.subplots()
cmap = plt.cm.Spectral
norm = plt.Normalize(vmin=V.min(), vmax=V.max())
ax.imshow(V, cmap=cmap, norm=norm, interpolation='nearest')

# Add the policy arrows
for i in range(policy.shape[0]):
    for j in range(policy.shape[1]):
        if policy[i, j]:
            if policy[i, j] == 'up':
                ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy[i, j] == 'down':
                ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy[i, j] == 'left':
                ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy[i, j] == 'right':
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
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label='State Value')
plt.show()


# Plot state values
plt.imshow(V, cmap='winter', interpolation='none')
plt.colorbar(label='State Value')

# Annotate each grid cell with the value
for x in range(V.shape[0]):
    for y in range(V.shape[1]):
        if gridworld[x, y] == 0:  # Only annotate non-terminal states
            plt.text(y, x, f'{V[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('State Value')
plt.show()


# Plot the returns over iterations
plt.plot(range(1, iterations + 1), returns)
plt.xlabel('Iteration')
plt.ylabel('Return (Sum of State Values)')
plt.title('Returns Over Iterations')
plt.show()
