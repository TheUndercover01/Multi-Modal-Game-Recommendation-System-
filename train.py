import torch
import numpy as np
import torch.optim as optim
from utils import create_padding_tensor
from utils import create_user_embeddings
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from actorcritic import create_agent
from test import TestDiscriminator
from tqdm import tqdm
import torch.nn.functional as f
import torch.distributions as distributions
from forward_pass import forward_pass
from losses import calculate_losses
from losses import calculate_surrogate_loss
from model.Generator import Generator
from model.Discriminator import Discriminator
from dataset import create_sequential_dataloader
from dataset import generate_dummy_data
from dataset import DictStateDataset
from utils import compute_action_prob
from forward_pass import _get_obs, get_state, step_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F



def update_policy(
        agent,
        states,
        actions,
        actions_log_probability_old,
        advantages,
        returns,
        g_optimizer,
        d_optimizer,
        ppo_steps,
        epsilon,
        entropy_coefficient,
        ):
    BATCH_SIZE = 4
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()

    all_games = torch.load('all_games.pt')
    K=10

    # Create a dataset that handles dictionary states

    # Create dataset and dataloader
    training_results_dataset = DictStateDataset(
        states,
        actions,
        actions_log_probability_old,
        advantages,
        returns
    )

    batch_dataset = DataLoader(
        training_results_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    for _ in range(ppo_steps):

        print(_, '_')
        gen_games = create_padding_tensor(batch_size=4)
        for batch_idx, (static_features, game_history, current_embeddings,
                        game_history_generated_games, step, actions,
                        actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):

            print(batch_idx,"batch_idx")
            # Forward pass with all state components
            for i in range(K):

                print(actions.shape, actions_log_probability_old.shape, advantages.shape, returns.shape , 'long')

                action_pred, value_pred, gen_games = agent(
                     static_features,
                     game_history,
                     gen_games,
                     game_history_generated_games,
                     i
                 )
                #

                action_prob = compute_action_prob(action_pred)
                #
                value_pred = value_pred.squeeze(-1)
                #action_prob = f.softmax(action_pred, dim=-1)
                probability_distribution_new = distributions.Categorical(action_prob)
                entropy = probability_distribution_new.entropy()
                #
                # # Estimate new log probabilities using old actions
                #print(actions.shape, probability_distribution_new, 'shapjsijahcwicn', action_prob.shape, action_normalized.shape)
                #
                actions_log_probability_new = probability_distribution_new.log_prob(actions[:, i])
                #
                surrogate_loss = calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    epsilon,
                    advantages,
                    i
                )

                policy_loss, value_loss = calculate_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred,
                    i
                )

                print(policy_loss, value_loss, 'losssssss')

                g_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)  # Need retain_graph since we're using gen_games multiple times
                g_optimizer.step()

                # Value network backward pass
                d_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)
                d_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

# def update_policy(
#         agent,
#         states,
#         actions,
#         actions_log_probability_old,
#         advantages,
#         returns,
#         optimizer,
#
#         ppo_steps,
#         epsilon,
#         entropy_coefficient):
#     BATCH_SIZE = 128
#     total_policy_loss = 0
#     total_value_loss = 0
#     actions_log_probability_old = actions_log_probability_old.detach()
#     actions = actions.detach()
#     training_results_dataset = TensorDataset(
#             states,
#             actions,
#             actions_log_probability_old,
#             advantages,
#             returns)
#     batch_dataset = DataLoader(
#             training_results_dataset,
#             batch_size=BATCH_SIZE,
#             shuffle=False)
#     for _ in range(ppo_steps):
#         for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
#
#             # if batch_idx == 0:
#             #     his_embed_ = states['game_history'].clone()
#             #     his_embed_ = torch.cat([his_embed_, gen_games], dim=1)
#             # get new log prob of actions for all input states
#             action_pred, value_pred, gen_games = agent(states[0],states[1],states[2], states[3], states[4])
#             value_pred = value_pred.squeeze(-1)
#             action_prob = f.softmax(action_pred, dim=-1)
#             probability_distribution_new = distributions.Categorical(
#                     action_prob)
#             entropy = probability_distribution_new.entropy()
#             # estimate new log probabilities using old actions
#             actions_log_probability_new = probability_distribution_new.log_prob(actions)
#             surrogate_loss = calculate_surrogate_loss(
#                     actions_log_probability_old,
#                     actions_log_probability_new,
#                     epsilon,
#                     advantages)
#             policy_loss, value_loss = calculate_losses(
#                     surrogate_loss,
#                     entropy,
#                     entropy_coefficient,
#                     returns,
#                     value_pred)
#             optimizer.zero_grad()
#             policy_loss.backward()
#             value_loss.backward()
#             optimizer.step()
#             total_policy_loss += policy_loss.item()
#             total_value_loss += value_loss.item()
#     return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def evaluate(batch_data, agent, fixed_discriminator):
    agent.eval()
    episode_reward = 0
    step = 0
    done = False

    # Initialize embeddings
    employee_embeddings = batch_data['static_features'].to(device).clone()
    his_embeddings = batch_data['game_history'].to(device).clone()
    gen_games = create_padding_tensor(batch_size=4)

    state = get_state(batch_data)

    while not done:
        with torch.no_grad():
            if step == 0:
                his_embed_ = his_embeddings.clone()
                his_embed_ = torch.cat([his_embed_, gen_games], dim=1)

            # Get predictions
            action_pred, value_pred, gen_games = agent(
                employee_embeddings,
                his_embeddings,
                gen_games,
                his_embed_,
                step
            )

            # Instead of sampling, we take the best action during evaluation
            action_prob = f.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)

            # Get reward from the discriminator
            reward = step( action.item(), value_pred, fixed_discriminator, employee_embeddings, gen_games)

            episode_reward += reward
            step += 1

            if step >= 9:
                done = True

            state = _get_obs(
                employee_embeddings,
                his_embeddings,
                gen_games,
                his_embed_,
                step
            )

    return episode_reward


def run_ppo(generator, trainable_discriminator, fixed_discriminator, data_loader):
    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    iterations = 1000
    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []


    #agent = ActorCritic(generator, trainable_discriminator)

    # generator = Generator(his_embeddings_shape, k, employee_dim, total_dim_out=his_embeddings_shape[2]).to(device)
    # trainable_discriminator = Discriminator(his_embeddings_shape, k, employee_dim).to(device)

    agent = create_agent(generator, trainable_discriminator)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(trainable_discriminator.parameters(), lr=0.0002)
    for iteration in tqdm(range(iterations), desc="Training Progress"):
        print(iteration)



        for batch_idx, batch_data in enumerate(data_loader):
            batch_size = 4

            # employee_embeddings = batch_data['static_features'].to(device).clone()
            # his_embeddings = batch_data['game_history'].to(device).clone()
            # gen_games = create_padding_tensor(batch_size=4)

        #env = GameRecommendationEnv(data_loader, trainable_discriminator)
            train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(batch_data, agent, g_optimizer, d_optimizer, DISCOUNT_FACTOR, fixed_discriminator)

            policy_loss, value_loss = update_policy(
                agent,
                states,
                actions,
                actions_log_probability,
                advantages,
                returns,
                g_optimizer,
                d_optimizer,
                PPO_STEPS,
                EPSILON,
                ENTROPY_COEFFICIENT)
            test_reward = evaluate(batch_data, agent, fixed_discriminator)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
            mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))


        if iteration % PRINT_INTERVAL == 0:
            print(f'Episode: {iteration:3} | \
                  Mean Train Rewards: {mean_train_rewards:3.1f} \
                  | Mean Test Rewards: {mean_test_rewards:3.1f} \
                  | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                  | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {iteration} episodes')
            break

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = pd.read_csv('user_tag.csv')
    employee_embeddings = torch.tensor(data.iloc[:, 1:].values.astype(np.float32))

    game_history = torch.load('user_game_embeddings.pt', weights_only=False)
    #print(len(game_history['76561197970982479']))
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
    run_ppo(
        generator,
        trainable_discriminator,
        fixed_discriminator,
        dataloader,



    )
    # fig = tracker.plot_metrics()
    # plt.show()

    # Save models
    torch.save(generator.state_dict(), 'generator_ppo.pth')
    torch.save(trainable_discriminator.state_dict(), 'trainable_discriminator_ppo.pth')

if __name__ == '__main__':
    main()