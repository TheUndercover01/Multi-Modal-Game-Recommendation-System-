import gym
from gym import spaces
import numpy as np
import torch
from train import update_input_tensors

import gym
from gym import spaces
import numpy as np
import torch


class GameRecommendationEnv(gym.Env):
    def __init__(self, discriminator, batch_seq_length=10):
        super().__init__()

        # Store user data components

        self.seq_length = seq_length

        # Initialize current embeddings (generated games)
        self.current_embeddings = torch.zeros((seq_length, 2304))
        self.current_step = 0

        # Set up discriminator for reward calculation
        self.discriminator = discriminator
        self.discriminator.eval()

        # Define spaces
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2304,))
        self.observation_space = spaces.Dict({
            'static_features': spaces.Box(low=-np.inf, high=np.inf, shape=self.static_features.shape),
            'game_history': spaces.Box(low=-np.inf, high=np.inf, shape=self.original_history.shape),
            'current_embeddings': spaces.Box(low=-np.inf, high=np.inf, shape=self.current_embeddings.shape)
        })

    def reset(self):
        """Reset environment for new episode"""
        self.current_embeddings = torch.zeros_like(self.current_embeddings)
        self.current_step = 0
        return self._get_obs()

    def step(self, action, user_data):
        """Execute one recommendation step"""
        # Store action in current embeddings
        self.current_embeddings[self.current_step] = torch.FloatTensor(action)
        self.current_step += 1

        self.static_features = user_data['static_features'].clone()
        self.original_history = user_data['game_history'].clone()

        # Create discriminator input
        full_history = torch.cat([self.original_history, self.current_embeddings], dim=0)

        # Calculate reward
        with torch.no_grad():
            reward = self.discriminator(
                self.static_features.unsqueeze(0),
                full_history.unsqueeze(0)
            ).item()

        # Check completion
        done = self.current_step >= self.seq_length

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """Return current observation"""
        return {
            'static_features': self.static_features.numpy(),
            'game_history': self.original_history.numpy(),
            'current_embeddings': self.current_embeddings.numpy()
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass