import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

###### MONTAINCAR DEMONSTRATION ######
# This code demonstrates how to use Function approximation with MountainCar environment, but used State aggregation for feature mapping.

# No need to discretize the state space, we will use function approximation
"""
for eg: Linear approximation
Q_value[state[0], state[1], action] => q^(s,a) = w.Tx(s,a) 
w = weight vector (parameters)
x(s,a) = feature vector for state–action pair
"""

POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07

# Normalize the state space to faster the learning process
def norm_state(state):
    """ Normalize the state to be btw -1 and 1
    x_norm = 2 * (x - min) / (max - min) - 1"""
    pos , vel = state
    pos_n = 2 * (pos - POS_MIN) / (POS_MAX - POS_MIN) - 1
    vel_n = 2 * (vel - VEL_MIN) / (VEL_MAX - VEL_MIN) - 1
    return np.array([pos_n, vel_n])


# Define a feature mapping from (position, velocity, action) → fixed-size vector.
def feature_mapping(state, action):
    pos_n, vel_n = norm_state(state)
    # polynomial + action one-hot encoding
    a0 = 1 if action == 0 else 0
    a1 = 1 if action == 1 else 0
    a2 = 1 if action == 2 else 0
    return np.array([
        1.0,                 # bias
        pos_n, vel_n,        # linear terms
        pos_n*vel_n,         # interaction
        pos_n**2, vel_n**2,  # quadratic
        a0, a1, a2           # action indicators
    ])


# -----------------------------
# Q-value with linear FA
# -----------------------------
def q_value(w, state, action):
    """Calculate the Q-value using linear function approximation."""
    return np.dot(w, feature_mapping(state, action))

def e_greedy_action(w, state, epsilon, env):
    # BUGFIX: removed stale unpacking of discretized state
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    qvals = [q_value(w, state, a) for a in range(env.action_space.n)]
    return int(np.argmax(qvals))

def watch_agent(env, w, episodes=3):
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()  # Show the environment
            action = np.argmax([q_value(w, state, a) for a in range(env.action_space.n)])
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep+1} total reward: {total_reward}")

    
def train(env, num_episodes = 5000, alpha = 1e-3, gamma = 0.99, epsilon = 0.1):
    """Train loops: as follows :
    1.Initialize Weight parameters.
    2.Loop over episodes:
       1. Reset env.
       3. Loop steps:
          1. Choose action using epsilon-greedy policy - policy improvement step
          2. Take action, observe next state and reward.
          4. Update Q-value using linear function approximation -> Semi-gradient SARSA update rule.
          5. Move to next state.
    3. Track total rewards for ploting"""

    num_features = len(feature_mapping(env.reset()[0], 0))  # Number of features
    w = np.zeros(num_features, dtype=np.float64)
    returns = []

    for episode in range(num_episodes):
        state, info = env.reset()
        action = e_greedy_action(w, state, epsilon, env)
        G = 0.0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            G += reward

            if terminated or truncated:
                # target = r  (no bootstrap)
                delta = reward - q_value(w, state, action)
                w += alpha * delta * feature_mapping(state, action)
                break

            next_action = e_greedy_action(w, next_state, epsilon, env)
            # Semi-gradient SARSA update rule
            # SARSA target: r + γ Q(s', a')
            # w = w + alpha * (q_ - q) * x
            target = reward + gamma * q_value(w, next_state, next_action)
            delta = target - q_value(w, state, action)
            w += alpha * delta * feature_mapping(state, action)

            state, action = next_state, next_action

        if (episode+1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            print(f"Episode {episode+1}, Avg return (last 100 eps): {avg_return:.2f}")

        returns.append(G)

        # epsilon decay
        epsilon = max(0.05, epsilon * 0.999)

    return w, returns 

def main():
    # For training speed, prefer render_mode=None; turn on "human" only to watch learned behavior
    env = gym.make('MountainCar-v0')  # render_mode=None to speed up 
    w, returns = train(env, num_episodes=5000, alpha=1e-3, gamma=0.99, epsilon=0.3)
    env.close()

    # Create a new env for watching
    render_env = gym.make('MountainCar-v0', render_mode="human")
    watch_agent(render_env, w, episodes=10)  # Watch the agent after training
    render_env.close()


if __name__ == "__main__":
    main()
