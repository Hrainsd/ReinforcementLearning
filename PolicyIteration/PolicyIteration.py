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
policy = np.full(gridworld.shape, 'right', dtype=object)

# Initialize Q-values dictionary
Q = {}
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        Q[(i, j)] = np.zeros(4)  # 4 actions: right, left, down, up

# Lists to store values and returns
value_iterations = []
returns = []

def policy_evaluation(policy, V, gridworld, gamma, epsilon):
    eval_iterations = 0  # Counter for number of value function updates
    while True:
        delta = 0
        for i in range(gridworld.shape[0]):
            for j in range(gridworld.shape[1]):
                if gridworld[i, j] in [-1, 1]:  # Skip terminal states
                    continue

                action = policy[i, j]
                if action == 'right':
                    ni, nj = i, j + 1
                elif action == 'left':
                    ni, nj = i, j - 1
                elif action == 'down':
                    ni, nj = i + 1, j
                elif action == 'up':
                    ni, nj = i - 1, j

                if 0 <= ni < gridworld.shape[0] and 0 <= nj < gridworld.shape[1]:
                    reward = gridworld[ni, nj]
                    new_value = reward + gamma * V[ni, nj]
                else:
                    new_value = -1  # if out of bounds, assume reward and value are -1

                delta = max(delta, abs(new_value - V[i, j]))
                V[i, j] = new_value

        eval_iterations += 1  # Count number of value function updates
        value_iterations.append(np.copy(V))
        returns.append(V.sum())

        if delta < epsilon:
            break

    return eval_iterations  # Return number of value function updates

def policy_improvement(policy, V, gridworld, gamma):
    policy_stable = True
    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] in [-1, 1]:  # Skip terminal states
                continue

            old_action = policy[i, j]

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

            policy[i, j] = best_action

            if old_action != best_action:
                policy_stable = False

    return policy_stable

iterations = 0
total_eval_iterations = 0  # To track total number of value function updates
while True:
    eval_iterations = policy_evaluation(policy, V, gridworld, gamma, epsilon)
    total_eval_iterations += eval_iterations  # Accumulate number of value function updates
    policy_stable = policy_improvement(policy, V, gridworld, gamma)
    iterations += 1
    if policy_stable:
        break

# Print Q-values
print("\nQ-values:")
actions = ['right', 'left', 'down', 'up']
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:  # Only print for non-terminal states
            q_values = Q[(i, j)]
            print(f"State ({i}, {j}): {q_values}")

print("State value:")
print(V)

print("Optimal policy:")
print(policy)

print(f"Total iterations: {iterations}")

print(f"Total value function updates: {total_eval_iterations}")

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
plt.plot(returns)
plt.xlabel('Value Function Update')
plt.ylabel('Return (Sum of State Values)')
plt.title('Returns Over Value Function Updates')
plt.show()
