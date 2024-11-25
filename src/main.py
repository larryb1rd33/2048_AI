import time
from src.board_utils import extract_board
from src.ai_logic import find_best_move_nn
from src.game_interface import GameInterface
import tensorflow as tf
import json


def __init__(self, config_path="../config/config.json"):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    self.move_delay = config.get("move_delay")  # Default to 0.5 seconds

def main(self):
    # Load the trained model
    model = tf.keras.models.load_model("models/model.h5")

    game = GameInterface()
    try:
        while True:
            # Extract the current board
            board = extract_board(game.driver)

            print("Current board:" + str(board))

            # Find the best move using the neural network
            best_move = find_best_move_nn(model, board)
            print(f"Best move: {best_move}")

            # Perform the best move
            game.perform_move(best_move)

            # Add a delay between moves
            time.sleep(self.move_delay)  # Wait for 0.5 seconds before the next move

    except Exception as e:
        print("Game over or error occurred:", e)
    finally:
        game.close()

if __name__ == "__main__":
    main()