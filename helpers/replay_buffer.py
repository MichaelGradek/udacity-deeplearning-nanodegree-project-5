import task as Task
import random
from collections import deque, namedtuple


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        # initialize Replay Buffer object
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        # add new experience to memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        # sample a given number of experiences from the memory
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        # return the length of memory
        return len(self.memory)
