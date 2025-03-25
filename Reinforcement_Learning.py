import numpy as np

# GridWorld configuration
grid_size = 4
num_states = grid_size * grid_size
gamma = 1.0
theta = 1e-4  # convergence threshold

# Initialize value function
V = np.zeros(num_states)

# Define actions: up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Helper functions
def get_state(row, col):
    return row * grid_size + col

def get_position(state):
    return divmod(state, grid_size)

def get_next_states(state):
    row, col = get_position(state)
    next_states = []
    for dr, dc in actions:
        new_row = row + dr
        new_col = col + dc
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            next_state = get_state(new_row, new_col)
        else:
            next_state = state  # stay in the same state (hit wall)
        next_states.append(next_state)
    return next_states

# Value Iteration
iteration = 0
while True:
    delta = 0
    V_new = np.copy(V)
    for s in range(num_states):
        if s == num_states - 1:
            continue  # Terminal state
        next_states = get_next_states(s)
        # Apply Bellman update: average over possible actions
        V_new[s] = sum([-1 + gamma * V[s_prime] for s_prime in next_states]) / 4.0
        delta = max(delta, abs(V_new[s] - V[s]))
    V = V_new
    iteration += 1
    if delta < theta:
        break

# Output
print(f"Converged in {iteration} iterations.")
print("Value Function:")
print(np.round(V.reshape((grid_size, grid_size)), 8))  # Show full precision
