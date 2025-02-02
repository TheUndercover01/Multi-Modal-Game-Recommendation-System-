
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F

def create_user_embeddings(user_data):
    """
    Transform nested user game data into a list of concatenated embeddings.
    Pads with -1 arrays if games count < 400.

    Args:
        user_data (dict): Nested dictionary of user game data
        Example structure:
        {
            'user1': [
                {
                    'image_embedding': embed1,
                    'title_embedding': embed2,
                    'about_embedding': embed3
                },
                {...}
            ],
            'user2': [...]
        }

    Returns:
        list: List of lists containing concatenated embeddings for each user, padded to 400 games
    """
    embeddings = []

    # Get the embedding dimension by concatenating the first game's embeddings
    first_user = list(user_data.values())[0]
    if first_user:
        first_game = first_user[0]
        sample_concat = concat_embeddings([
            first_game['image_embedding'],
            first_game['title_embedding'],
            first_game['about_embedding']
        ])
        padding_dim = len(sample_concat)

    for user, games in user_data.items():
        user_embeddings = []

        # Process existing games
        for game in games:
            game_concat = concat_embeddings([
                game['image_embedding'],
                game['title_embedding'],
                game['about_embedding']
            ])
            # print(game['game_title'])
            user_embeddings.append(game_concat)

        # Pad with -1 arrays if needed
        num_games = len(games)
        if num_games < 400:
            padding = [-1 * np.ones(padding_dim) for _ in range(400 - num_games)]
            user_embeddings.extend(padding)

        embeddings.append(user_embeddings)

    return embeddings


def compute_action_prob(action_pred, index = False):
    all_games = torch.load('all_games.pt')
    action_normalized = F.normalize(action_pred, p=2, dim=1)
    all_games = F.normalize(all_games, p=2, dim=1)
    similarities = torch.mm(action_normalized, all_games.t())

    action_prob = F.softmax(similarities, dim=1)
    if index:
        max_similarities, max_indices = torch.max(action_prob, dim=1)
        return action_prob, max_indices
    return action_prob



def concat_embeddings(embed_list):
    """
    Concatenate a list of embeddings.
    """
    con = np.concatenate(embed_list, axis=1)
    return con[0]


def create_padding_tensor(batch_size=1):
    """
    Creates a tensor for 10 games with embeddings of length 2304 filled with -1s,
    supporting multiple batches.

    Args:
        batch_size: Number of batches to create

    Returns:
        torch.Tensor: Shape (batch_size, 10, 2304) filled with -1s
    """
    # Create padding tensor with batch dimension
    padding_tensor = -1 * np.ones((batch_size, 10, 2304))
    padding_games = torch.tensor(padding_tensor, dtype=torch.float32)
    return padding_games



class SequentialGANDataset(Dataset):
    def __init__(self,
                 static_features,  # (N, 449) initial static features
                 initial_games,  # (N, 400, 2304) initial game sequence
                 game_embeddings,  # (N, 10, 2304) initial game embeddings
                 target_games_seq):  # (N, K, 410, 2304) K target games for each sample
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
        self.game_embeddings = torch.FloatTensor(game_embeddings)
        self.target_games_seq = torch.FloatTensor(target_games_seq)

        # Verify shapes
        self.verify_shapes()

    def verify_shapes(self):
        N = len(self.static_features)
        K = self.target_games_seq.shape[1]

        assert self.static_features.shape == (
        N, 449), f"Static features should be (N, 449), got {self.static_features.shape}"
        assert self.initial_games.shape == (
        N, 400, 2304), f"Initial games should be (N, 400, 2304), got {self.initial_games.shape}"
        assert self.game_embeddings.shape == (
        N, 10, 2304), f"Game embeddings should be (N, 10, 2304), got {self.game_embeddings.shape}"
        assert self.target_games_seq.shape[2:] == (
        410, 2304), f"Target games should have shape (N, K, 410, 2304), got {self.target_games_seq.shape}"

    def __len__(self):
        return len(self.static_features)

    def __getitem__(self, idx):
        return {
            'input1': self.static_features[idx],
            'input2': self.initial_games[idx],
            'input3': self.game_embeddings[idx],
            'target_games': self.target_games_seq[idx]
        }


def create_sequential_dataloader(
        static_features,
        initial_games,
        game_embeddings,
        target_games_seq,
        batch_size=32,
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
        game_embeddings,
        target_games_seq
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# Example usage:
def generate_dummy_data(num_samples=4, seq_length=10):
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


# Usage example

    # Generate dummy data

def update_input_tensors(new_game, current_input2, current_input3, step):
    """
    Update input tensors with newly generated game

    Args:
        new_game: Generated game tensor (batch_size, 2304)
        current_input2: Current game sequence (batch_size, N, 2304)
        current_input3: Current game embeddings (batch_size, 10, 2304)
        step: Current step in sequence (0-based index)
    """
    batch_size = new_game.size(0)

    # Update input3 by replacing the game at position 'step'
    updated_input3 = current_input3.clone()
    updated_input3[:, step] = new_game

    updated_input2 = current_input2.clone()
    updated_input2[:, 400+ step] = new_game

    # Update input2 by appending the new game
    # updated_input2 = torch.cat([
    #     current_input2,
    #     new_game.unsqueeze(1)
    # ], dim=1)

    return updated_input2, updated_input3