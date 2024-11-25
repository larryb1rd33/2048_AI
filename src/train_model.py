import logging
from src.models.replay_buffer import ReplayBuffer
from src.game_interface import GameInterface
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    filename="training_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Hyperparameters
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPISODES = 1000
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
epsilon_min = 0.01
WORKERS = 8  # Number of parallel environments

# Initialize the model
model = tf.keras.models.load_model("../src/models/model.h5")  # Policy or Q-network
replay_buffer = ReplayBuffer(max_size=10000)

logging.info("🚀 Starting Training...")

def run_episode(env_id):
    """
    Run a single episode in a parallel environment and return collected data.
    """
    logging.info(f"Environment {env_id} - Starting Episode")
    env = GameInterface()
    state = np.array(env.reset()).flatten()
    episode_data = []  # To store transitions for replay buffer
    total_reward = 0
    steps = 0

    while True:
        # Choose action (explore vs exploit)
        if np.random.random() < epsilon:
            action = np.random.choice([0, 1, 2, 3])  # Represent actions as integers
        else:
            q_values = model.predict(state[np.newaxis], verbose=0)  # Exploit
            action = np.argmax(q_values)

        # Map action index to direction
        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        env_action = action_map[action]

        # Take action and observe result
        next_state, reward, done = env.step(env_action)
        episode_data.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            logging.info(f"Environment {env_id} - Episode Complete")
            logging.info(f"  - Total Reward: {total_reward}")
            logging.info(f"  - Steps Taken: {steps}")
            env.close()
            break

    return episode_data

# Parallel execution of episodes
for episode in range(EPISODES):
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(run_episode, i) for i in range(WORKERS)]

        for future in futures:
            episode_data = future.result()
            for state, action, reward, next_state, done in episode_data:
                replay_buffer.add(state, action, reward, next_state, done)

    # Train the model from replay buffer
    if len(replay_buffer) > BATCH_SIZE:
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        # Compute target Q-values
        target_q = model.predict(states, verbose=0)
        next_q = model.predict(next_states, verbose=0)
        for i in range(BATCH_SIZE):
            target_q[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q[i]) * (1 - dones[i])

        # Update the model
        history = model.fit(states, target_q, verbose=0)

        # Log loss and training progress
        logging.info(f"Episode {episode} - Loss: {history.history['loss'][-1]:.4f}")
        logging.info(f"Episode {episode} - Batch Q-values: {np.mean(target_q):.4f}")

    # Decay epsilon after each episode
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Save the final model
model.save("../src/models/trained_model.h5")
logging.info("🎉 Training Complete! Final model saved to 'trained_model.h5'")