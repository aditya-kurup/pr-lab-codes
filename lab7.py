import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += p @ p.T
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                s = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if s >= 0 else -1
        return pattern


def print_pattern(p, shape):
    p = p.reshape(shape)
    for row in p:
        print(''.join('1' if x == 1 else '0' for x in row))
    print()


# Example patterns (5x5)
pattern_A = np.array([
    -1, 1, 1, 1, -1,
     1,-1,-1,-1, 1,
     1, 1, 1, 1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1
])

pattern_B = np.array([
     1, 1, 1, 1,-1,
     1,-1,-1,-1, 1,
     1, 1, 1, 1,-1,
     1,-1,-1,-1, 1,
     1, 1, 1, 1,-1
])

# Train
net = HopfieldNetwork(25)
net.train([pattern_A, pattern_B])

# Noisy input
noisy = pattern_A.copy()
noisy[0] = 1
noisy[6] = 1

print("Noisy:")
print_pattern(noisy, (5,5))

# Recall
recovered = net.recall(noisy)

print("Recovered:")
print_pattern(recovered, (5,5))