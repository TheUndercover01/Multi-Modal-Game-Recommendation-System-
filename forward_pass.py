import torch
from utils import create_padding_tensor
from actorcritic import calculate_returns
from actorcritic import calculate_advantages
import torch.nn.functional as f
import torch.distributions as distributions
from utils import compute_action_prob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    return states, actions, actions_log_probability, values, rewards, done, episode_reward

def get_state(batch_data, his_embed_, step):
    dict =  {
        'static_features': batch_data['static_features'].to(device).numpy(),
        'game_history': batch_data['game_history'].to(device).numpy(),
        'current_embeddings': create_padding_tensor(batch_size=4),
        'game_history_generated_games': his_embed_,
        'step': [step]
    }
    # tensors = [
    #     torch.tensor(dict['static_features']),
    #     torch.tensor(dict['game_history']),
    #     torch.tensor(dict['current_embeddings']),
    #     torch.tensor(dict['game_history_generated_games']),
    #     torch.tensor(dict['step'])]

    return dict


def _get_obs(employee_embeddings, history_embeddings, gen_games, his_embed_, step):
    """Return current observation"""
    dict =  {
        'static_features': employee_embeddings.numpy(),
        'game_history': history_embeddings.numpy(),
        'current_embeddings': gen_games.detach().numpy(),
        'game_history_generated_games': his_embed_,
        'step': [step]
    }

    # tensors = [
    # torch.tensor(dict['static_features']),
    # torch.tensor(dict['game_history']),
    # torch.tensor(dict['current_embeddings']),
    # torch.tensor(dict['game_history_generated_games']),
    # torch.tensor(dict['step'])]

    return dict
def step_(value_pred, fixed_discriminator,  employee_embeddings, gen_games):
    with torch.no_grad():  # Don't track gradients for value estimation during collection
        #value = trainable_discriminator(employee_embeddings, his_dis)
        reward = fixed_discriminator(employee_embeddings, gen_games)

    #composite reward
    reward +=  value_pred


    return reward




def forward_pass(batch_data, agent, g_optimizer, d_optimizer, discount_factor, fixed_discriminator):
    # Initialize lists for each component of the state
    states_static = []
    states_history = []
    states_current = []
    states_gen_games = []
    states_steps = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    episode_reward = 0



    employee_embeddings = batch_data['static_features'].to(device).clone()
    his_embeddings = batch_data['game_history'].to(device).clone()
    gen_games = create_padding_tensor(batch_size=4)
    agent.train()
    step = 0
    done = False

    while step < 10:
        if step == 0:
            his_embed_ = his_embeddings.clone()
            his_embed_ = torch.cat([his_embed_, gen_games], dim=1)
            state = get_state(batch_data, his_embed_, step)
        else:
            state =  _get_obs(employee_embeddings, his_embeddings, gen_games, his_embed_, step)

        action_pred, value_pred, gen_games = agent(employee_embeddings, his_embeddings, gen_games, his_embed_, step)

        # Append each component of the state separately
        states_static.append(torch.tensor(state['static_features']))
        states_history.append(torch.tensor(state['game_history']))
        states_current.append(torch.tensor(state['current_embeddings']))
        states_gen_games.append(torch.tensor(state['game_history_generated_games']))
        states_steps.append(torch.tensor(state['step']))

        print(torch.tensor(state['step']), 'step', step)

        action_prob = compute_action_prob(action_pred)


        print(action_prob, 'action_prob')
        dist = distributions.Categorical(action_prob)

        action = dist.sample()


        print(action, 'action')
        log_prob_action = dist.log_prob(action)
        print(log_prob_action, 'log_prob_action')
        reward = step_(value_pred, fixed_discriminator, employee_embeddings, gen_games)
        print(reward, 'reward')
        #state = _get_obs(employee_embeddings, his_embeddings, gen_games, his_embed_, step)



        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

        step += 1
        # if step >=9:
        #     done = True

    # Concatenate each component separately
    states = {
        'static_features': torch.cat(states_static),
        'game_history': torch.cat(states_history),
        'current_embeddings': torch.cat(states_current),
        'game_history_generated_games': torch.cat(states_gen_games),
        'step': torch.cat(states_steps)
    }

    print(states['step'] , 'stepsdskmc')

    actions = torch.stack(actions, dim=1)

    print(actions.shape, 'action')
    actions_log_probability = torch.stack(actions_log_probability, dim=1)
    print(actions_log_probability.shape, 'action_log_prob')
    values = torch.stack(values, dim=1)
    print(values.shape, 'value')
    returns = calculate_returns(rewards, discount_factor)
    print(returns.shape, 'rewards')
    advantages = calculate_advantages(returns, values)
    print(advantages.shape, 'advantages')

    return episode_reward, states, actions, actions_log_probability, advantages, returns