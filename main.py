import torch
from utils import create_user_embeddings,create_user_embeddings_1
from model.Generator import Generator
from model.Discriminator import Discriminator
from torch.utils.data import DataLoader
from test import TestDiscriminator
from dataset import create_sequential_dataloader
from train import run_ppo
import pandas as pd
import numpy as np


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = pd.read_csv('user_tag.csv')
    employee_embeddings = torch.tensor(data.iloc[:, 1:].values.astype(np.float32))

    game_history = torch.load('user_game_embeddings.pt', weights_only=False)
    #print(len(game_history['76561197970982479']))
    his_embeddings = torch.tensor(create_user_embeddings_1(game_history), dtype=torch.float32)



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
    #fixed_discriminator = Discriminator(his_embeddings_shape, k, employee_dim).to(device)

    # Load fixed discriminator weights (assuming pre-trained)
    fixed_discriminator= torch.load("model.pth")

    fixed_discriminator.eval()
    # fixed_discriminator = TestDiscriminator(his_embeddings_shape, k, employee_dim).to(device)

    # Training parameters
    iterations = 10
    K = 10  # steps per sequence

    #static_features, initial_games, game_embeddings, target_games_seq = generate_dummy_data()
    print(1)

    # Create dataloader
    dataloader = create_sequential_dataloader(
        employee_embeddings,
        his_embeddings,
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
    print("Finished")
if __name__ == '__main__':
    main()
