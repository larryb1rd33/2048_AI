from src.board_utils import extract_board
from src.game_interface import GameInterface
import numpy as np
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    filename="../src/training_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_training_data(num_games, task_id):
    """
    Generate training data using random moves or heuristic evaluations.
    Each task generates a portion of the data.
    """
    X = []  # Board states
    y = []  # Move scores

    for game_index in range(num_games):
        logging.info(f"Task {task_id}: 🎮 Starting Game {game_index + 1}/{num_games}")

        game = GameInterface()
        move_count = 0  # Track the number of moves in the game
        previous_boards = []  # Track previous board states for exit condition

        try:
            while True:
                move_count += 1

                # Extract the current board
                board = extract_board(game.driver)
                flattened_board = np.array(board).flatten()
                X.append(flattened_board)

                # Simulate moves and generate scores (replace random scores with heuristics if needed)
                move_scores = [random.random() for _ in range(4)]
                y.append(move_scores)

                # Fetch the current score
                current_score = game.get_current_score()

                # Perform a random move
                moves = ["up", "down", "left", "right"]
                chosen_move = random.choice(moves)
                game.perform_move(chosen_move)

                # Track previous boards and check for unchanged state
                if len(previous_boards) == 5:
                    previous_boards.pop(0)  # Remove the oldest board state
                previous_boards.append(board)

                if previous_boards.count(board) == 5:
                    logging.info(f"Task {task_id}: ⚠️ Board unchanged for 5 consecutive rounds. Ending game.")
                    break

                # Pause briefly to simulate human-like interaction
                time.sleep(0.2)

        except Exception as e:
            logging.error(f"Task {task_id}: ❌ Exception occurred: {e}")
        finally:
            game.close()
            logging.info(
                f"Task {task_id}: 📌 Game {game_index + 1} Summary: Moves: {move_count}, Final Score: {current_score}")

    return np.array(X), np.array(y)


def main():
    num_games = 250  # Total number of games
    num_tasks = 8  # Number of concurrent tasks
    games_per_task = num_games // num_tasks  # Divide games evenly among tasks

    logging.info("\n🚀 Generating Training Data with Concurrency...\n")

    # Use ThreadPoolExecutor for concurrency
    X_data = []
    y_data = []
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [
            executor.submit(generate_training_data, games_per_task, task_id)
            for task_id in range(num_tasks)
        ]

        for future in as_completed(futures):
            try:
                X, y = future.result()
                X_data.append(X)
                y_data.append(y)
            except Exception as e:
                logging.error(f"❌ Error in one of the tasks: {e}")

    # Combine data from all tasks
    X_combined = np.vstack(X_data)
    y_combined = np.vstack(y_data)

    # Save the combined data
    np.save("../src/X.npy", X_combined)
    np.save("../src/y.npy", y_combined)

    logging.info("\n✅ Training Data Generation Complete!")
    logging.info(f"📁 Data saved to X.npy and y.npy")
    logging.info(f"🎮 Total Games Played: {num_games}")
    logging.info(f"🔢 Total Data Points Collected: {len(X_combined)}\n")


if __name__ == "__main__":
    main()