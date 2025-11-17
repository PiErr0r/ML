import random

import numpy as np
from activation_functions import sigmoid, sigmoid_prime

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.act_fn = sigmoid
        self.act_fn_prime = sigmoid_prime

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = w @ a + b
        return a

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)
        n_test = 0 if test_data is None else len(test_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            for k in range(0, n, mini_batch_size):
                mini_batch = training_data[k:k + mini_batch_size]
                self.SGD(mini_batch, eta)

            if not test_data is None:
                E = self.evaluate(test_data)
                print(f"Epoch {epoch}: {E}/{n_test}")
            else:
                print(f"Epoch {epoch} complete")

    def SGD(self, data, eta):
        m = len(data)
        nabla_b, nabla_w = self.backprop(data)
        self.biases = [(b.T - (eta / m) * dnb.sum(axis=1)).T for b, dnb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / m) * dnw for w, dnw in zip(self.weights, nabla_w)]

    def backprop(self, data):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        x, y = zip(*data)
        x = np.array(x).transpose(2, 1, 0)[0]
        y = np.array(y).transpose(2, 1, 0)[0]

        activations = [x]
        activation = x
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = w @ activation + b
            zs.append(z)
            activation = self.act_fn(z)
            activations.append(activation)

        delta = self.cost_derivative(y, activations[-1]) * self.act_fn_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T
        for l in range(2, self.num_layers):
            prev = -l + 1
            nxt = -l - 1
            delta = (self.weights[prev].T @ delta) * self.act_fn_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[nxt].T

        return nabla_b, nabla_w


    def cost_derivative(self, y, a):
        return a - y

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                         for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

