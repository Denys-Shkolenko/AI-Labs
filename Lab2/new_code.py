import json

import numpy as np
from tabulate import tabulate


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        if hidden_layers:
            self.weights.append(np.random.randn(input_size, hidden_layers[0]))
            self.biases.append(np.zeros(hidden_layers[0]))

            for i in range(1, len(hidden_layers)):
                self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]))
                self.biases.append(np.zeros(hidden_layers[i]))

            self.weights.append(np.random.randn(hidden_layers[-1], output_size))
            self.biases.append(np.zeros(output_size))
        else:
            self.weights.append(np.random.randn(input_size, output_size))
            self.biases.append(np.zeros(output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        a = x
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
        return a

    def train(self, x, y, learning_rate, epochs):
        for _ in range(epochs):
            a = x
            activations = [a]
            zs = []

            for i in range(len(self.weights)):
                z = np.dot(a, self.weights[i]) + self.biases[i]
                zs.append(z)
                a = self.sigmoid(z)
                activations.append(a)

            delta = self.cost_derivative(activations[-1], y) * self.sigmoid_derivative(zs[-1])

            for i in range(len(self.weights) - 1, -1, -1):
                gradient_weights = np.dot(activations[i].T, delta)
                gradient_biases = np.sum(delta, axis=0)

                self.weights[i] -= learning_rate * gradient_weights
                self.biases[i] -= learning_rate * gradient_biases

                if i > 0:
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(zs[i-1])

    def predict(self, x):
        return self.forward(x)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


if __name__ == "__main__":
    with open("training_data.json") as training_data:
        training_data = json.load(training_data)

    X = []
    y = []
    answers = {}

    for data in training_data:
        X.append(data['symbol'])
        y.append(data['8-bit code'])
        answers[tuple(data['8-bit code'])] = data['answer']

    X = np.array(X)
    y = np.array(y)

    epochs = 1000
    learning_rate = 0.1

    with open("tests.json") as training_data:
        test_data = json.load(training_data)

    input_size = 36
    hidden_layers = [10, 8]
    output_size = 3

    nn = NeuralNetwork(input_size, hidden_layers, output_size)

    nn.train(X, y, learning_rate, epochs)

    table = []
    print("*" * 15, "TESTS", "*" * 15)
    for data in test_data:
        input_data = np.array([data['symbol']])
        expected_output = np.array([data['8-bit code']])

        predicted_output = nn.predict(input_data)
        predicted_output_rounded = np.round(predicted_output, 2)

        table.append([
            data["answer"],
            np.array2string(expected_output, separator=' ')[1:-1],
            np.array2string(predicted_output_rounded, separator=' ')[1:-1]
        ])

    print(tabulate(table, headers=["Image", "Expected Value", "y"], tablefmt="orgtbl"))

    input_size = 36
    hidden_layers = []
    output_size = 3

    nn2 = NeuralNetwork(input_size, hidden_layers, output_size)

    nn2.train(X, y, learning_rate, epochs)

    table = []
    print("*" * 15, "TESTS", "*" * 15)
    for data in test_data:
        input_data = np.array([data['symbol']])
        expected_output = np.array([data['8-bit code']])

        predicted_output = nn2.predict(input_data)
        predicted_output_rounded = np.round(predicted_output, 2)

        table.append([
            data["answer"],
            np.array2string(expected_output, separator=' ')[1:-1],
            np.array2string(predicted_output_rounded, separator=' ')[1:-1]
        ])

    print(tabulate(table, headers=["Image", "Expected Value", "y"], tablefmt="orgtbl"))



