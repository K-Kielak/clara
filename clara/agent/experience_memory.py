import random
import numpy as np
from collections import deque


class Memory(object):
    EXPERIENCE_SIZE = 4

    def __init__(self, memory_size):
        self.experiences = deque()
        self.memory_size = memory_size
        self.samples_given = 0

    def add(self, initial_state, action, reward, following_state):
        experience = np.array([initial_state, action, reward, following_state])
        self.experiences.append(experience)

        if len(self.experiences) > self.memory_size:
            self.experiences.popleft()

    def get_samples(self, sample_size):
        samples = random.sample(self.experiences, sample_size)
        try:
            self.samples_given += 1
            return np.reshape(samples, [sample_size, Memory.EXPERIENCE_SIZE])
        except ValueError:
            print(self.samples_given)
            print(len(samples))
            for s in samples:
                try:
                    print(len(s))
                    print(len(s[0]))
                    print(s[1])
                    print(s[2])
                    print(len(s[3]))
                except IndexError:
                    print(s)
