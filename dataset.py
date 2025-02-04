import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SequentialGANDataset(Dataset):
    def __init__(self,
                 static_features,  # (N, 449) initial static features
                 initial_games,  # (N, 400, 2304) initial game sequence
                 #game_embeddings,  # (N, 10, 2304) initial game embeddings
                 ):  # (N, K, 410, 2304) K target games for each sample
        """
        Dataset for sequential GAN training

        Args:
            static_features: Initial static features (N, 449)
            initial_games: Initial game sequence (N, 400, 2304)
            game_embeddings: Initial game embeddings (N, 10, 2304)
            target_games_seq: Target games sequence used for discriminator training (N, K, 410, 2304)
        """
        # Convert inputs to tensors if they're not already
        self.static_features = torch.FloatTensor(static_features)
        self.initial_games = torch.FloatTensor(initial_games)
        #self.game_embeddings = torch.FloatTensor(game_embeddings)
        #self.target_games_seq = torch.FloatTensor(target_games_seq)

        # Verify shapes
        self.verify_shapes()

    def verify_shapes(self):
        N = len(self.static_features)
        K = 10

        assert self.static_features.shape == (
        N, 449), f"Static features should be (N, 449), got {self.static_features.shape}"
        assert self.initial_games.shape == (
        N, 400, 2304), f"Initial games should be (N, 400, 2304), got {self.initial_games.shape}"
        #assert self.game_embeddings.shape == (
        #N, 10, 2304), f"Game embeddings should be (N, 10, 2304), got {self.game_embeddings.shape}"
        #assert self.target_games_seq.shape[2:] == (
        #410, 2304), f"Target games should have shape (N, K, 410, 2304), got {self.target_games_seq.shape}"

    def __len__(self):
        return len(self.static_features)

    def __getitem__(self, idx):
        return {
            'static_features': self.static_features[idx],
            'game_history': self.initial_games[idx],
            #'generated_games': self.game_embeddings[idx],
            #'target_games': self.target_games_seq[idx]
        }


def create_sequential_dataloader(
        static_features,
        initial_games,
        batch_size=4,
        shuffle=True,
        num_workers=4
):
    """
    Creates a DataLoader for sequential GAN training.

    Args:
        static_features: numpy array/tensor (N, 449)
        initial_games: numpy array/tensor (N, 400, 2304)
        game_embeddings: numpy array/tensor (N, 10, 2304)
        target_games_seq: numpy array/tensor (N, K, 410, 2304)
        batch_size: int, batch size for training
        shuffle: bool, whether to shuffle the data
        num_workers: int, number of worker processes

    Returns:
        DataLoader object
    """
    dataset = SequentialGANDataset(
        static_features,
        initial_games,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# Example usage:
def generate_dummy_data(num_samples=50, seq_length=10):
    """
    Generate dummy data for testing the dataloader

    Args:
        num_samples: number of samples to generate
        seq_length: number of sequential steps (K)
    """
    static_features = np.random.randn(num_samples, 449)
    initial_games = np.random.randn(num_samples, 400, 2304)
    game_embeddings = np.random.randn(num_samples, 10, 2304)
    target_games_seq = np.random.randn(num_samples, seq_length, 410, 2304)

    return static_features, initial_games, game_embeddings, target_games_seq

class DictStateDataset(Dataset):
    def __init__(self, states_dict, actions, actions_log_prob, advantages, returns):
        self.static_features = states_dict['static_features']
        self.game_history = states_dict['game_history']
        self.current_embeddings = states_dict['current_embeddings']
        self.game_history_generated_games = states_dict['game_history_generated_games']
        self.step = states_dict['step']
        self.actions = actions
        self.actions_log_prob = actions_log_prob
        self.advantages = advantages
        self.returns = returns

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.static_features[idx],
            self.game_history[idx],
            self.current_embeddings[idx],
            self.game_history_generated_games[idx],
            self.step[idx],
            self.actions[idx],
            self.actions_log_prob[idx],
            self.advantages[idx],
            self.returns[idx]
        )



# Usage example
if __name__ == "__main__":
    # Generate dummy data
    static_features, initial_games, game_embeddings, target_games_seq = generate_dummy_data()
    print(1)

    # Create dataloader
    dataloader = create_sequential_dataloader(
        static_features,
        initial_games,
        game_embeddings,
        target_games_seq,
        batch_size=32
    )

    # Example iteration
    for batch in dataloader:
        input1 = batch['input1']  # (batch_size, 449)
        input2 = batch['input2']  # (batch_size, 400, 2304)
        input3 = batch['input3']  # (batch_size, 10, 2304)
        target_games = batch['target_games']  # (batch_size, K, 410, 2304)

        # Your training code here
        break  # Remove this line when actually training