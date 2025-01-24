

import torch
import numpy as np

import pandas as pd
from Generator import Generator
from Discriminator import Discriminator
import torch.nn as nn


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

# print([0]*10)
# Example usage:
gen_games = create_padding_tensor(batch_size = 4)

# Verify the shape and values
#print("Shape:", gen_games.shape)
#print("Sample value check:", gen_games)
print(gen_games.shape)
data = pd.read_csv('user_tag.csv')
    #get the employee embeddings
employee_embeddings = data.iloc[:, 1:].values

employee_embeddings = employee_embeddings.astype(np.float32)
employee_embeddings = torch.tensor(employee_embeddings , dtype=torch.float32)
# print("slidfnhvcsldjnwoef" , employee_embeddings.shape)

game_history = torch.load('user_game_embeddings.pt', weights_only=False)

#print(game_history)



#print(game_history['doctr'])
embeddings = create_user_embeddings(game_history)

his_embeddings = torch.tensor(embeddings , dtype=torch.float32)

print("his", his_embeddings.shape)

k = 10
his_embeddings_shape = his_embeddings.shape

employee_dim = employee_embeddings.shape[1]  # Assuming the third dimension is the embedding dimension
print(employee_embeddings.shape)
# print(his_embeddings.max() , his_embeddings.min() , his_embeddings.shape)


generator = Generator(his_embeddings_shape, k, employee_dim, total_dim_out = his_embeddings_shape[2])

discriminator = Discriminator(his_embeddings_shape, k, employee_dim)

game_generated = generator(employee_embeddings, his_embeddings, gen_games)

historical_gen = torch.cat((his_embeddings , gen_games) ,  1 )
print(historical_gen.shape)

score = discriminator(employee_embeddings , historical_gen)

print('score' , score)

# print(game_generated.max() , game_generated.min() , game_generated.shape)
# #torch.save(game_generated, 'generated_games.pt')
#
print('game generated' , game_generated)