import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from Generator import Generator
from Discriminator import Discriminator
from main import create_padding_tensor
from main import create_user_embeddings
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from test import TestDiscriminator
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()


def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        next_value = 0 if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lambda_ * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return torch.stack(returns)


class MetricsTracker:
    def __init__(self):
        self.actor_losses = []
        self.critic_losses = []
        self.rewards = []
        self.iteration_summaries = []

    def log(self, actor_loss, critic_loss, rewards):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.rewards.append(rewards.mean().item())

    def log_iteration_summary(self, iteration, avg_loss, avg_reward):
        summary = {
            'iteration': iteration,
            'avg_loss': avg_loss,
            'avg_reward': avg_reward,
            'actor_loss_avg': sum(self.actor_losses[-100:]) / min(len(self.actor_losses), 100),
            'critic_loss_avg': sum(self.critic_losses[-100:]) / min(len(self.critic_losses), 100),
            'reward_avg': sum(self.rewards[-100:]) / min(len(self.rewards), 100)
        }
        self.iteration_summaries.append(summary)
        return summary

    def plot_metrics(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        window_size = 100  # For smoothing

        # Plot with moving average for smoother visualization
        def moving_average(data, window):
            return [sum(data[max(0, i - window):i]) / min(i, window) for i in range(1, len(data) + 1)]

        ax1.plot(moving_average(self.actor_losses, window_size))
        ax1.set_title('Actor Loss (Moving Avg)')
        ax1.set_xlabel('Steps')

        ax2.plot(moving_average(self.critic_losses, window_size))
        ax2.set_title('Critic Loss (Moving Avg)')
        ax2.set_xlabel('Steps')

        ax3.plot(moving_average(self.rewards, window_size))
        ax3.set_title('Average Reward (Moving Avg)')
        ax3.set_xlabel('Steps')

        plt.tight_layout()
        return fig


def ppo_training_loop(generator, trainable_discriminator, fixed_discriminator,
                      data_loader, iterations, K, device, clip_epsilon=0.2, c1=1.0, c2=0.01):
    memory = PPOMemory()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(trainable_discriminator.parameters(), lr=0.0002)
    tracker = MetricsTracker()

    for iteration in tqdm(range(iterations), desc="Training Progress"):
        batch_losses = []
        batch_rewards = []

        for batch_idx, batch_data in enumerate(data_loader):
            batch_size = 4
            # Create fresh copies of inputs to prevent in-place modifications
            employee_embeddings = batch_data['input1'].to(device).clone()
            his_embeddings = batch_data['input2'].to(device).clone()
            gen_games = create_padding_tensor(batch_size=4)

            # Collect trajectories
            trajectory_states = []
            trajectory_actions = []
            trajectory_log_probs = []
            trajectory_values = []
            trajectory_rewards = []

            for step in range(K):
                # Generate action probabilities
                with torch.no_grad():  # Don't track gradients during sampling
                    action_probs = generator(employee_embeddings, his_embeddings, gen_games)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    action_log_prob = dist.log_prob(action)

                # Generate new game
                new_game = generator(employee_embeddings, his_embeddings, gen_games)

                # Handle history embedding
                if step == 0:
                    his_embed_ = his_embeddings.clone()
                    his_embed_ = torch.cat([his_embed_, gen_games], dim=1)

                # Update game state - create new tensors instead of in-place updates
                his_dis, new_gen_games = update_input_tensors(new_game, his_embed_, gen_games, step)
                gen_games = new_gen_games.clone()  # Use new tensor instead of in-place update

                # Compute value and reward
                with torch.no_grad():  # Don't track gradients for value estimation during collection
                    value = trainable_discriminator(employee_embeddings, his_dis)
                    reward = fixed_discriminator(employee_embeddings, his_dis)

                # Store trajectory information
                trajectory_states.append(action_probs.detach())  # Detach to prevent gradient tracking
                trajectory_actions.append(action)
                trajectory_log_probs.append(action_log_prob)
                trajectory_values.append(value)
                trajectory_rewards.append(reward)

            # Convert trajectories to tensors
            states = torch.stack(trajectory_states)
            actions = torch.stack(trajectory_actions)
            old_log_probs = torch.stack(trajectory_log_probs)
            values = torch.stack(trajectory_values)
            rewards = torch.stack(trajectory_rewards)

            # Compute returns and advantages
            returns = compute_gae(rewards, values, torch.zeros_like(rewards))
            advantages = returns - values.detach()

            # PPO Update
            for _ in range(1):  # PPO epochs
                # Generate new predictions
                new_values = trainable_discriminator(employee_embeddings, his_dis)#states_batch)
                new_action_probs = generator(employee_embeddings, his_embeddings, gen_games)
                new_dist = torch.distributions.Categorical(new_action_probs)
                new_log_probs = new_dist.log_prob(actions)

                # Compute PPO losses
                print(_, "PPO losses")
                ratios = torch.exp(new_log_probs - old_log_probs.detach())
                print(ratios.shape , "rat", advantages.shape)

                surr1 = ratios * advantages.squeeze(-1).detach()  # Detach advantages
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages.squeeze(-1).detach()

                actor_loss = torch.min(surr1, surr2).mean()#-
                critic_loss = F.mse_loss(new_values, returns)
                entropy_loss = -c2 * new_dist.entropy().mean()

                total_loss = -actor_loss + c1 * critic_loss + entropy_loss

                # Optimization step
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                g_optimizer.step()
                d_optimizer.step()

                tracker.log(actor_loss.item(), critic_loss.item(), rewards[-1])  # Log final reward
                batch_losses.append(total_loss.item())
                batch_rewards.append(rewards[-1].mean().item())

        # Log iteration summary
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        summary = tracker.log_iteration_summary(iteration, avg_loss, avg_reward)
        print(summary)

    return tracker


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
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
    generator = Generator(his_embeddings_shape, k, employee_dim, total_dim_out=his_embeddings_shape[2]).to(device)
    trainable_discriminator = Discriminator(his_embeddings_shape, k, employee_dim).to(device)
    fixed_discriminator = Discriminator(his_embeddings_shape, k, employee_dim).to(device)

    # Load fixed discriminator weights (assuming pre-trained)
    # fixed_discriminator.load_state_dict(torch.load('fixed_discriminator.pth'))
    # fixed_discriminator.eval()
    fixed_discriminator = TestDiscriminator(his_embeddings_shape, k, employee_dim).to(device)

    # Training parameters
    iterations = 10
    K = 10  # steps per sequence

    static_features, initial_games, game_embeddings, target_games_seq = generate_dummy_data()
    print(1)

    # Create dataloader
    dataloader = create_sequential_dataloader(
        static_features,
        initial_games,
        game_embeddings,
        target_games_seq,
        batch_size=4
    )

    # Start training
    tracker = ppo_training_loop(
        generator,
        trainable_discriminator,
        fixed_discriminator,
        dataloader,
        iterations,
        K,
        device,
        clip_epsilon=0.2,
        c1=1.0,
        c2=0.01
    )
    fig = tracker.plot_metrics()
    plt.show()

    # Save models
    torch.save(generator.state_dict(), 'generator_ppo.pth')
    torch.save(trainable_discriminator.state_dict(), 'trainable_discriminator_ppo.pth')


if __name__ == '__main__':
    main()