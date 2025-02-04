import torch

import torch.nn as nn
class TestDiscriminator(nn.Module):
    def __init__(self, his_embeddings_shape, k, employee_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(employee_dim + his_embeddings_shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, employee_embeddings, his_embeddings):
        # Use last game in history
        last_game = his_embeddings[:, -1]
        combined = torch.cat([employee_embeddings, last_game], dim=1)
        return self.network(combined)

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

    # Update input2 by appending the new game
    updated_input2 = torch.cat([
        current_input2,
        new_game.unsqueeze(1)
    ], dim=1)

    return updated_input2, updated_input3


def test_sequential_replacement():
    print("Testing Sequential Replacement Pattern")
    print("=" * 50)

    # Setup test data
    batch_size = 2
    game_dim = 4  # Using smaller dimension for clearer output
    sequence_length = 10
    initial_games = 3  # Using smaller number for clearer output

    # Initialize input tensors with position-indicating values
    current_input2 = torch.zeros((batch_size, initial_games, game_dim))
    current_input3 = torch.zeros((batch_size, sequence_length, game_dim))

    # Fill input3 with position-indicating values
    for pos in range(sequence_length):
        current_input3[:, pos] = pos + 1

    print("Initial input3 (showing first batch, first feature):")
    print(current_input3[0, :, 0])  # Show first feature of first batch
    print("\nStarting step-by-step replacement test...")

    # Test for first 5 steps
    for step in range(5):
        print(f"\nStep {step}")
        print("-" * 50)

        # Create new game with distinctive value for this step
        new_game_value = 99 + step  # e.g., 99, 100, 101, etc.
        new_game = torch.ones((batch_size, game_dim)) * new_game_value

        # Store original values for verification
        original_step_value = current_input3[0, step, 0].item()

        print(f"Original value at position {step}: {original_step_value}")
        print(f"New game value to insert: {new_game_value}")

        # Update tensors
        updated_input2, updated_input3 = update_input_tensors(
            new_game,
            current_input2,
            current_input3,
            step
        )

        # Verify replacement
        print("\nValues in input3 after update (first batch, first feature):")
        print(updated_input3[0, :, 0])

        # Detailed verification
        print("\nVerification:")
        print(f"Position {step} should now contain {new_game_value}")
        assert torch.allclose(updated_input3[:, step], new_game), \
            f"Position {step} not correctly updated"

        # Verify other positions remained unchanged
        for pos in range(sequence_length):
            if pos != step:
                original_value = current_input3[0, pos, 0].item()
                updated_value = updated_input3[0, pos, 0].item()
                print(f"Position {pos}: {'unchanged' if original_value == updated_value else 'CHANGED!'} "
                      f"({original_value} -> {updated_value})")
                assert original_value == updated_value, \
                    f"Position {pos} changed unexpectedly"

        # Update current tensors for next iteration
        current_input2 = updated_input2
        current_input3 = updated_input3


if __name__ == "__main__":
    try:
        test_sequential_replacement()
        print("\nAll tests passed successfully! ✓")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")