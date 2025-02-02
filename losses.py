import torch

import torch.nn.functional as f

def calculate_surrogate_loss(
        actions_log_probability_old,
        actions_log_probability_new,
        epsilon,
        advantages, step):
    advantages = advantages.detach()
    policy_ratio = (
            actions_log_probability_new - actions_log_probability_old[: , step]
            ).exp()
    surrogate_loss_1 = policy_ratio * advantages[: step]
    surrogate_loss_2 = torch.clamp(
            policy_ratio, min=1.0-epsilon, max=1.0+epsilon
            ) * advantages[: step]
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss


def calculate_losses(
        surrogate_loss, entropy, entropy_coefficient, returns, value_pred, step):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).sum()

    print(returns[: , step] )
    value_loss = f.smooth_l1_loss(returns[:, step], value_pred).sum()
    return policy_loss, value_loss