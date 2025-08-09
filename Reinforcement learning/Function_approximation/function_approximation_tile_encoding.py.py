import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


###### MONTAINCAR DEMONSTRATION ######
# This code demonstrates how to use Function approximation with MountainCar environment, but used tile encoding instead of state aggregation for feature mapping.

# -----------------------------
# Tile coding helper functions
# -----------------------------
def create_tilings(low, high, num_tilings, bins, offsets):
    """
    Create multiple overlapping tilings for a continuous space.
    Each tiling is defined by bin boundaries and an offset.
    
    Args:
        low, high: min/max of each dimension (arrays)
        num_tilings: number of tilings
        bins: list with number of bins per dimension
        offsets: list with offset fractions per tiling
    
    Returns:
        tilings: list of bin boundaries for each tiling
    """
    tilings = []
    for tiling_i in range(num_tilings):
        tiling_bins = []
        for dim in range(len(low)):
            # Shift boundaries by an offset
            offset = offsets[tiling_i][dim]
            boundaries = np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offset
            tiling_bins.append(boundaries)
        tilings.append(tiling_bins)
    return tilings

def tile_encode(state, tilings):
    """
    Encode a continuous state into a sparse binary feature vector.
    
    Args:
        state: continuous state (pos, vel)
        tilings: list of tiling boundaries
    
    Returns:
        features: binary vector where active tiles = 1
    """
    features = []
    for tiling in tilings:
        indices = [np.digitize(s, boundaries) for s, boundaries in zip(state, tiling)]
        features.append(tuple(indices))
    return features

def tile_features(state, action, tilings, bins_per_dim, num_actions):
    """
    Convert (state, action) into a 1D binary feature vector for linear FA.
    """
    features = np.zeros((len(tilings) * np.prod(bins_per_dim) * num_actions,))
    tile_indices = tile_encode(state, tilings)
    for tiling_idx, tile_coord in enumerate(tile_indices):
        # Compute index for this tile in this tiling
        tile_index = np.ravel_multi_index(tile_coord, bins_per_dim)
        # Offset index for tiling & action
        index = tiling_idx * np.prod(bins_per_dim) * num_actions + tile_index * num_actions + action
        features[index] = 1
    return features

# -----------------------------
# Q-value with linear FA
# -----------------------------
def q_value(w, state, action, tilings, bins_per_dim, num_actions):
    return np.dot(w, tile_features(state, action, tilings, bins_per_dim, num_actions))

def e_greedy_action(w, state, epsilon, env, tilings, bins_per_dim, num_actions):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    qvals = [q_value(w, state, a, tilings, bins_per_dim, num_actions) for a in range(num_actions)]
    return int(np.argmax(qvals))

# -----------------------------
# Watching the agent
# -----------------------------
def watch_agent(env, w, episodes, tilings, bins_per_dim, num_actions):
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = e_greedy_action(w, state, 0.0, env, tilings, bins_per_dim, num_actions)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep+1} total reward: {total_reward}")

# -----------------------------
# SARSA(0) with tile coding
# -----------------------------
def train(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Semi-gradient SARSA with linear FA using tile coding.
    """
    num_actions = env.action_space.n
    bins_per_dim = (8, 8)   # tiles per dimension
    num_tilings = 8         # number of overlapping tilings

    # Create small random offsets for each tiling
    offsets = [ (np.random.uniform(0, 1) * (POS_MAX - POS_MIN) / (bins_per_dim[0]*num_tilings),
                 np.random.uniform(0, 1) * (VEL_MAX - VEL_MIN) / (bins_per_dim[1]*num_tilings))
                for _ in range(num_tilings) ]
    
    tilings = create_tilings([POS_MIN, VEL_MIN], [POS_MAX, VEL_MAX], num_tilings, bins_per_dim, offsets)

    # Weight vector size = total tiles × num_actions
    num_tiles_total = num_tilings * np.prod(bins_per_dim) * num_actions
    w = np.zeros(num_tiles_total)

    returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = e_greedy_action(w, state, epsilon, env, tilings, bins_per_dim, num_actions)
        G = 0.0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            G += reward

            if terminated or truncated:
                delta = reward - q_value(w, state, action, tilings, bins_per_dim, num_actions)
                w += alpha * delta * tile_features(state, action, tilings, bins_per_dim, num_actions)
                break

            next_action = e_greedy_action(w, next_state, epsilon, env, tilings, bins_per_dim, num_actions)
            target = reward + gamma * q_value(w, next_state, next_action, tilings, bins_per_dim, num_actions)
            delta = target - q_value(w, state, action, tilings, bins_per_dim, num_actions)
            w += alpha * delta * tile_features(state, action, tilings, bins_per_dim, num_actions)

            state, action = next_state, next_action

        returns.append(G)

        if (episode+1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            print(f"Episode {episode+1}, Avg return (last 100 eps): {avg_return:.2f}")

        epsilon = max(0.05, epsilon * 0.999)

    return w, returns, tilings, bins_per_dim, num_actions

# -----------------------------
# Main
# -----------------------------
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07

def main():
    env = gym.make('MountainCar-v0')  # Train without rendering
    w, returns, tilings, bins_per_dim, num_actions = train(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.3)
    env.close()

    render_env = gym.make('MountainCar-v0', render_mode="human")
    watch_agent(render_env, w, episodes=10, tilings=tilings, bins_per_dim=bins_per_dim, num_actions=num_actions)
    # render_env.close()

if __name__ == "__main__":
    main()
