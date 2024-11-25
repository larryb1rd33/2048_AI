import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import logging

def create_model():
    """
    Create a neural network model for 2048 with a specified learning rate.
    """
    logging.info("Creating a new neural network model.")
    model = Sequential([
        Input(shape=(16,)),  # Input layer: 16 neurons for the board state
        Dense(128, activation='relu'),  # Hidden layer: 128 neurons
        Dense(64, activation='relu'),  # Hidden layer: 64 neurons
        Dense(4, activation='linear')  # Output layer: 4 neurons for the moves
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Specify the learning rate
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    logging.info("Model created successfully.")
    return model


def find_best_move_nn(model, board):
    """
    Use the neural network to find the best move.
    Args:
        model: Trained TensorFlow model.
        board: 2D list representing the game board.

    Returns:
        The best move as a string ("up", "down", "left", "right").
    """
    logging.info("Evaluating the best move using the neural network.")

    # Flatten the board state
    input_state = np.array(board).flatten().reshape(1, 16)
    logging.info(f"Flattened Board State: {input_state.tolist()}")

    # Predict move scores
    move_scores = model.predict(input_state, verbose=0)[0]
    logging.info(f"Predicted Move Scores: {move_scores.tolist()}")

    # Map moves to scores
    moves = ["up", "down", "left", "right"]
    best_move_index = np.argmax(move_scores)
    best_move = moves[best_move_index]
    logging.info(f"Best Move: {best_move} with Score: {move_scores[best_move_index]}")

    return best_move