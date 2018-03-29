import random


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, mol):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = mol
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)