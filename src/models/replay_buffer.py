import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        # Ensure `action` is an integer
        self.buffer.append((state, int(action), reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions, dtype=int),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)