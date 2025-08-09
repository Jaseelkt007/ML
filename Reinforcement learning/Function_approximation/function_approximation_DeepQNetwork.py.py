import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

###### MONTAINCAR DEMONSTRATION ######
# This code demonstrates how to use Function approximation using Deep Q - Network with MountainCar environment, which uses raw state value as input, no need of feature mapping.

# -----------------------------
# DQN Network Definition
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # A simple 2-hidden-layer MLP for MountainCar (state_dim=2)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # One output per action
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Epsilon-greedy action selection
# -----------------------------
def select_action(state, policy_net, epsilon, env, device):
    if random.random() < epsilon:
        return env.action_space.sample()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Training Loop
# -----------------------------
def train_dqn(env, num_episodes=500, gamma=0.99, lr=1e-3,
              batch_size=64, buffer_size=50000,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
              target_update_freq=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device) # Policy network
    target_net = DQN(state_dim, action_dim).to(device) # Target network to stabilize the training
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = select_action(state, policy_net, epsilon, env, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Sample from buffer and train
            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)

                states_b = torch.FloatTensor(states_b).to(device)
                actions_b = torch.LongTensor(actions_b).unsqueeze(1).to(device)
                rewards_b = torch.FloatTensor(rewards_b).to(device)
                next_states_b = torch.FloatTensor(next_states_b).to(device)
                dones_b = torch.FloatTensor(dones_b).to(device)

                # Current Q values
                q_values = policy_net(states_b).gather(1, actions_b)

                # Target Q values
                with torch.no_grad():
                    max_next_q = target_net(next_states_b).max(1)[0]
                    target_q = rewards_b + gamma * max_next_q * (1 - dones_b)

                loss = nn.MSELoss()(q_values.squeeze(), target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        episode_rewards.append(total_reward)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network periodically
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode+1) % 10 == 0:
            avg_last_10 = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}, Avg Reward (last 10): {avg_last_10:.2f}, Epsilon: {epsilon:.3f}")

    return policy_net, episode_rewards, device

# -----------------------------
# Watch the trained agent
# -----------------------------
def watch_agent(env, policy_net, device, episodes=3):
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = select_action(state, policy_net, 0.0, env, device)  # Greedy
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep+1} total reward: {total_reward}")

# -----------------------------
# Main
# -----------------------------
def main():
    env = gym.make('MountainCar-v0')
    policy_net, rewards, device = train_dqn(env, num_episodes=500)
    env.close()

    # Watch the trained agent
    render_env = gym.make('MountainCar-v0', render_mode="human")
    watch_agent(render_env, policy_net, device, episodes=10)
    render_env.close()

if __name__ == "__main__":
    main()
