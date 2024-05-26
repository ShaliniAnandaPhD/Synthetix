#Model-Based Reinforcement Learning for Contact-Rich Manipulation" (2022)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Define the ManipulationDataset class to handle data loading
class ManipulationDataset(Dataset):
    def __init__(self, data_dir):
        # Initialize the dataset
        self.data_dir = data_dir
        self.data = self.load_data()

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get an item from the dataset at the given index
        state, action, next_state, reward = self.data[idx]
        return state, action, next_state, reward

    def load_data(self):
        # Load the data from the data directory
        # Assumes the data is stored in numpy files: 'states.npy', 'actions.npy', 'next_states.npy', 'rewards.npy'
        states = np.load(os.path.join(self.data_dir, 'states.npy'))
        actions = np.load(os.path.join(self.data_dir, 'actions.npy'))
        next_states = np.load(os.path.join(self.data_dir, 'next_states.npy'))
        rewards = np.load(os.path.join(self.data_dir, 'rewards.npy'))
        data = [(states[i], actions[i], next_states[i], rewards[i]) for i in range(len(states))]
        return data

# Define the DynamicsModel class
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DynamicsModel, self).__init__()
        # Define the model architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        # Concatenate the state and action
        x = torch.cat([state, action], dim=-1)
        # Pass through the model layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state_pred = self.fc3(x)
        return next_state_pred

# Define the RewardModel class
class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RewardModel, self).__init__()
        # Define the model architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concatenate the state and action
        x = torch.cat([state, action], dim=-1)
        # Pass through the model layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        reward_pred = self.fc3(x)
        return reward_pred

# Set the device to use (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the data directory
data_dir = 'path/to/your/data/directory'

# Create an instance of the ManipulationDataset
dataset = ManipulationDataset(data_dir)

# Create a DataLoader to handle batching and shuffling
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set the model hyperparameters
state_dim = dataset.data[0][0].shape[0]  # Assumes states are 1D arrays
action_dim = dataset.data[0][1].shape[0]  # Assumes actions are 1D arrays
hidden_dim = 256

# Create instances of the DynamicsModel and RewardModel
dynamics_model = DynamicsModel(state_dim, action_dim, hidden_dim).to(device)
reward_model = RewardModel(state_dim, action_dim, hidden_dim).to(device)

# Set the loss functions and optimizers
dynamics_criterion = nn.MSELoss()
reward_criterion = nn.MSELoss()
dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=0.001)
reward_optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

# Set the number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    running_dynamics_loss = 0.0
    running_reward_loss = 0.0
    for states, actions, next_states, rewards in dataloader:
        # Move the data to the device
        states = states.float().to(device)
        actions = actions.float().to(device)
        next_states = next_states.float().to(device)
        rewards = rewards.float().to(device)

        # Zero the parameter gradients
        dynamics_optimizer.zero_grad()
        reward_optimizer.zero_grad()

        # Forward pass
        next_state_preds = dynamics_model(states, actions)
        reward_preds = reward_model(states, actions)
        # Compute the losses
        dynamics_loss = dynamics_criterion(next_state_preds, next_states)
        reward_loss = reward_criterion(reward_preds, rewards)

        # Backward pass and optimization
        dynamics_loss.backward()
        reward_loss.backward()
        dynamics_optimizer.step()
        reward_optimizer.step()

        # Update the running losses
        running_dynamics_loss += dynamics_loss.item()
        running_reward_loss += reward_loss.item()

    # Print the average losses for the epoch
    epoch_dynamics_loss = running_dynamics_loss / len(dataloader)
    epoch_reward_loss = running_reward_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Dynamics Loss: {epoch_dynamics_loss:.4f}, Reward Loss: {epoch_reward_loss:.4f}")

# Save the trained models
torch.save(dynamics_model.state_dict(), 'dynamics_model.pth')
torch.save(reward_model.state_dict(), 'reward_model.pth')

# Possible Errors and Solutions:
# 1. Error: FileNotFoundError when loading the dataset.
#    Solution: Ensure that the `data_dir` variable points to the correct directory containing the 'states.npy',
#              'actions.npy', 'next_states.npy', and 'rewards.npy' files.
#
# 2. Error: ValueError: too many values to unpack (expected 4).
#    Solution: Make sure that each item in the dataset consists of a tuple of (state, action, next_state, reward).
#              Check the `load_data` method and ensure that the data is structured correctly.
#
# 3. Error: RuntimeError: size mismatch, m1: [batch_size x state_dim], m2: [batch_size x (state_dim + action_dim)].
#    Solution: Verify that the dimensions of the states and actions match the expected dimensions in the models.
#              Adjust the `state_dim` and `action_dim` variables accordingly.
#
# 4. Error: Poor manipulation performance.
#    Solution: Experiment with different model architectures, such as using deeper or wider networks, or incorporating
#              recurrent layers for sequential data. Increase the size and diversity of the dataset to cover various
#              manipulation scenarios. Fine-tune the hyperparameters and consider using techniques like data augmentation
#              or regularization.

# Note: The code assumes a specific dataset format where the states, actions, next states, and rewards are stored in
#       separate numpy files. Adapt the `load_data` method according to your dataset's format and file names.

# To use the trained models for manipulation:
# 1. Load the saved model weights using `dynamics_model.load_state_dict(torch.load('dynamics_model.pth'))` and
#    `reward_model.load_state_dict(torch.load('reward_model.pth'))`.
# 2. Obtain the current state of the environment.
# 3. Select an action based on a planning algorithm or policy (e.g., model predictive control, reinforcement learning).
# 4. Use the dynamics model to predict the next state given the current state and selected action.
# 5. Use the reward model to predict the reward for the current state and selected action.
# 6. Execute the selected action in the environment and observe the actual next state and reward.
# 7. Update the models using the observed data if desired (online learning).
# 8. Repeat steps 3-7 for the desired number of timesteps or until a termination condition is met.
