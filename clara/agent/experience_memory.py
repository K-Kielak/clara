import random
from collections import deque


class Memory(object):
    def __init__(self, memory_size):
        self.experiences = deque()
        self.memory_size = memory_size

    def add(self, experience):
        if len(self.experiences) < self.memory_size:
            self.experiences.append(experience)
            return

        self.experiences.popleft()
        self.experiences.append(experience)

    def get_samples(self, sample_size):
        return random.sample(self.experiences, sample_size)
