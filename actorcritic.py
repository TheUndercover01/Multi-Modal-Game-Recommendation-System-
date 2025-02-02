import torch
import torch.nn as nn
import numpy as np
from model.Generator import Generator
from model.Discriminator import Discriminator
from utils import create_user_embeddings
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as f
from utils import update_input_tensors
from utils import compute_action_prob



class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.all_games = torch.load('all_games.pt')
    def forward(self, employee_embeddings, his_embeddings, gen_games, his_embed_, step):
        action_pred = self.actor(employee_embeddings, his_embeddings, gen_games) #get game
        #get real game from action_pred
         #switch action_pred with real game
        action_prob, max_indices = compute_action_prob(action_pred, index=True)

        # Get max similarity values and indices
        #max_similarities, max_indices = torch.max(similarities, dim=1)




        # Update game state - create new tensors instead of in-place updates
        his_dis, new_gen_games = update_input_tensors(self.all_games[max_indices], his_embed_, gen_games, step)
        gen_games = new_gen_games.clone()

        value_pred = self.critic(employee_embeddings, his_dis) #get score
        return action_pred, value_pred, gen_games


def calculate_returns(rewards, discount_factor):
    if not rewards:
        raise ValueError("Rewards list cannot be empty")

    # Initialize returns list with the first reward tensor's shape
    returns = []
    cumulative_reward = torch.zeros_like(rewards[0])

    # Calculate returns
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)

    # Stack the tensors along a new dimension
    returns = torch.stack(returns, dim = 1)  # This will give shape [time_steps, batch_size, 1]
    print("Return", returns.shape, 'actorcritic')

    # Normalize along the time dimension (dim=0)
    if len(returns) > 1:
        mean = returns.mean(dim=0, keepdim=True)
        std = returns.std(dim=0, keepdim=True) + 1e-8
        returns = (returns - mean) / std

    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    # Normalize the advantage
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages


def create_agent(generator, discriminator):
    data = pd.read_csv('user_tag.csv')
    employee_embeddings = torch.tensor(data.iloc[:, 1:].values.astype(np.float32))

    game_history = torch.load('user_game_embeddings.pt', weights_only=False)
    his_embeddings = torch.tensor(create_user_embeddings(game_history), dtype=torch.float32)

    # Setup parameters
    batch_size = 32
    k = 10  # sequence length
    his_embeddings_shape = his_embeddings.shape
    employee_dim = employee_embeddings.shape[1]

    # Create dataset and dataloader
    # dataset = CustomDataset(employee_embeddings, his_embeddings)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    actor = Generator(his_embeddings_shape, k, employee_dim, total_dim_out=his_embeddings_shape[2]).to(device)


    critic = Discriminator(his_embeddings_shape, k, employee_dim).to(device)
    agent = ActorCritic(actor, critic)
    return agent

