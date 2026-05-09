---
title: "Rl"
date: "2025-04-30T19:00:15+08:00"
draft: false
---

```python
class QNetwork(nn.Module):
  def __init__(self, env):
    super(QNetwork, self).__init__()
    # Get observation space dimensions (expected to be 8: 4 for state, 2 for pos_ee, 2 for goal)
    self.obs_dim = env.observation_space.shape[0]

    # Define a discrete action space
    # Let's create a grid of torque values for each joint
    # For example, for each joint: [-0.1, -0.05, 0, 0.05, 0.1]
    # This gives us 5*5=25 possible action combinations for 2 joints
    self.torque_values = [-0.5, -0.25, 0, 0.25, 0.5]
    self.num_discrete_actions = len(self.torque_values) ** 2

    # Create action mapping dictionary for converting discrete to continuous
    self.action_mapping = {}
    idx = 0
    for t1 in self.torque_values:
        for t2 in self.torque_values:
            self.action_mapping[idx] = np.array([t1, t2])
            idx += 1

    # Neural network architecture
    # Input layer: observation dimension
    # Hidden layers: Two layers with 128 and 64 units
    # Output layer: Number of discrete actions
    self.fc1 = nn.Linear(self.obs_dim, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, self.num_discrete_actions)
    # 这样的层级架构可以避免128，128，128的过拟合问题，出现损失，reward降低不下去，很有可能就是过拟合了

  def forward(self, x, device):
    # Convert numpy array to tensor if necessary
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)

    # Move input tensor to specified device
    x = x.to(device)

    # Forward pass through the network
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))  
    x = self.fc4(x)          

    return x
```

```python
from stable_baselines3.ppo import PPO
import os
import time
from stable_baselines3.common.utils import set_random_seed


# Default parameters
timesteps = 500000
nenv = 8  # number of parallel environments. This can speed up training when you have good CPUs
seed = 8
batch_size = 2048

# Generate path of the directory to save the checkpoint
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join('ppo_models', timestr)

# Set random seed
set_random_seed(seed)

# Create arm
arm = make_arm()

# Create parallel envs
vec_env = make_vec_env(arm=arm, nenv=nenv, seed=seed)

# ------ IMPLEMENT YOUR TRAINING CODE HERE ------------

# Create the PPO model
model = PPO(
    policy='MlpPolicy',
    env=vec_env,
    batch_size=batch_size,
    seed=seed,
    verbose=1,
    n_epochs=40,         # 每次更新的训练轮数
    # 邪招，10 -> 40 极大的增加了reward
    learning_rate=3e-4,  # 学习率
    gamma=0.99,          # 折扣因子
    gae_lambda=0.95,     # GAE lambda
    clip_range=0.2,      # 策略裁剪范围
)


# Train the model
model.learn(total_timesteps=timesteps)

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the trained model
model.save(os.path.join(save_dir, 'ppo_network'))
```

