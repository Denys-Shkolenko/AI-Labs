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

    INPUT_DATA = (2.57, 4.35, 1.27, 5.46, 1.30, 4.92, 1.31,
                  4.14, 1.97, 5.67, 0.92, 4.76, 1.72, 4.44, 1.49)

    X = []
    y = []

    for i in range(len(INPUT_DATA) - 3):
        X.append([INPUT_DATA[i],
                  INPUT_DATA[i + 1],
                  INPUT_DATA[i + 2]])
        y.append([INPUT_DATA[i + 3]])

    X = np.array(X)
    y = np.array(y)
    # print(X)
    # print(y)
    epochs = 1000
    learning_rate = 0.9

    nn = NeuralNetwork(3, [3], 1)
    nn.train(X, y, learning_rate, epochs)

    # Вхідні дані
    INPUT_DATA = [2.57, 4.35, 1.27, 5.46, 1.30, 4.92, 1.31, 4.14, 1.97, 5.67, 0.92, 4.76, 1.72, 4.44, 1.49]

    # Тестування
    print("*" * 15, "TESTS", "*" * 15)
    table = []
    for i in range(0, len(INPUT_DATA), 4):
        # Отримуємо вхідні дані і очікуване значення для поточного набору
        input_data = np.array([INPUT_DATA[i:i + 3]])
        expected_output = np.array([INPUT_DATA[i + 3]])

        # Передбачаємо значення з використанням нейронної мережі
        predicted_output = nn.predict(input_data)

        # Додаємо результати до таблиці
        table.append([
            input_data,
            expected_output,
            predicted_output
        ])

    # Виводимо таблицю результатів
    headers = ["Input", "Expected Value", "Predicted Value"]
    print(tabulate(table, headers=headers, tablefmt="orgtbl"))
