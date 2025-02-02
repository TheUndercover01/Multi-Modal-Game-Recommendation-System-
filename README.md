# Multi-Modal-Game-Recommendation-System-


This project implements a game recommendation system using Proximal Policy Optimization (PPO) with a generator-discriminator architecture. The system learns to recommend games based on user preferences and gaming history.

## Data Requirements

The system requires two main data inputs:

1. `user_tag.csv`: Contains gamer personal embeddings
   
2. `user_game_embeddings.pt`: PyTorch file containing user game history
   - Dictionary format with user IDs as keys
   - Each value contains the user's gaming history embeddings
   - Used to understand user's gaming patterns and preferences

## Setup

1. Install required dependencies:
```bash
pip install torch pandas numpy tqdm
```

2. Place your data files in the project root:
   - `user_tag.csv`
   - `user_game_embeddings.pt`
   - `all_games.pt` - Here all 60,000 games must be mapped or stored (database of game embeddings)

## Project Structure

- `main.py`: Main training script
- `model/`: Neural network architectures
  - `Generator.py`: Game recommendation generator
  - `Discriminator.py`: Reward discriminator
- `utils`: Helper functions
- `dataset.py`: Data loading and processing
- `losses.py`: Loss function implementations
- `forward_pass.py`: Forward pass logic for PPO

## Running the Code

To train the model:

```bash
python main.py
```

The training process will:
1. Load user embeddings and game history
2. Initialize generator and discriminator networks
3. Train using PPO algorithm
4. Save trained models as `generator_ppo.pth` and `trainable_discriminator_ppo.pth`

## Model Outputs

The system generates game recommendations in sequences of 10 games, optimizing for user preferences and gaming patterns.

## Parameters

- Batch size: 4
- Sequence length (K): 10
- PPO steps: 8
- Learning rate: 0.0002
- Max episodes: 500
- Discount factor: 0.99
```

