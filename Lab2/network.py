from operator import itemgetter

import numpy as np
from typing import Sequence

from functions import derivative_err_func


class Network:

    def __init__(self,
                 size_of_input_layer=36,
                 sizes_of_hidden_layers=(36, 30),
                 size_of_output_layer=3,
                 default_weight=0.1,
                 learning_rate=0.1,
                 eps=0.0001,
                 max_iterations=1_000_000,
                 activation_func=derivative_err_func):

        if not isinstance(size_of_input_layer, int):
            raise TypeError("size_of_input_layer must be an int")
        if size_of_input_layer < 1:
            raise ValueError("size_of_input_layer must be greater than 0")
        self.__size_of_input_layer = size_of_input_layer

        if not all(isinstance(size, int) for size in sizes_of_hidden_layers):
            raise TypeError("sizes_of_hidden_layers must be integers")
        if any(size < 1 for size in sizes_of_hidden_layers):
            raise ValueError("sizes_of_hidden_layers must be greater than 0")
        self.__sizes_of_hidden_layers = sizes_of_hidden_layers

        if not isinstance(size_of_output_layer, int):
            raise TypeError("size_of_output_layer must be an int")
        if size_of_output_layer < 1:
            raise ValueError("size_of_output_layer must be greater than 0")
        self.__size_of_output_layer = size_of_output_layer

        self.__number_of_hidden_layers = len(sizes_of_hidden_layers)
        self.__number_of_layers = self.__number_of_hidden_layers + 2
        self.__sizes = [size_of_input_layer, *sizes_of_hidden_layers, size_of_output_layer]
        self.__s_of_layers = [[0] for _ in range(self.__number_of_hidden_layers + 1)]

        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iterations = max_iterations
        self.activation_func = activation_func

        # initialization of weights
        self.weights = []
        sizes = (size_of_input_layer, *sizes_of_hidden_layers, size_of_output_layer)
        for rows, cols in zip(sizes, sizes[1:]):
            matrix = np.full((rows, cols), default_weight)
            self.weights.append(matrix)

    @property
    def size_of_input_layer(self):
        return self.__size_of_input_layer

    @property
    def sizes_of_hidden_layers(self):
        return self.__sizes_of_hidden_layers

    @property
    def size_of_output_layer(self):
        return self.__size_of_output_layer

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")
        if not 0.0 < learning_rate < 1.0:
            raise ValueError("learning_rate must be in range (0; 1)")
        self.__learning_rate = learning_rate

    @property
    def eps(self):
        return self.__eps

    @eps.setter
    def eps(self, eps: float):
        if not isinstance(eps, float):
            raise TypeError("eps must be a float")
        self.__eps = eps

    @property
    def max_iterations(self):
        return self.__max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations: int):
        if not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an int")
        if max_iterations < 1:
            raise ValueError("max_iterations must be greater than 0")
        self.__max_iterations = max_iterations

    @property
    def number_of_hidden_layers(self):
        return self.__number_of_hidden_layers

    def set_input_data(self, input_data: Sequence[int]):
        if not input_data:
            raise ValueError("input_data cannot be empty")
        if len(input_data) != self.size_of_input_layer:
            raise ValueError("the len of input_data must be equal to the "
                             "size_of_input_layer parameter")
        if not all(isinstance(number, int) for number in input_data):
            raise TypeError("each item of the input_data must be an int")
        self.__s_of_layers[0] = input_data

    def set_expected_values(self, expected_values: Sequence[float]):
        if not expected_values:
            raise ValueError("expected_values cannot be empty")
        if len(expected_values) != self.size_of_output_layer:
            raise ValueError("the len of expected_values must be equal to the "
                             "size_of_output_layer parameter")
        if not all(isinstance(number, int) for number in expected_values):
            raise TypeError("each item of the input_data must be an int")
        self.__expected_values = expected_values

    def start_training(self) -> int:
        iterations = 0
        accuracy = self.eps + 1.0  # to start the loop
        while abs(accuracy) > self.eps and iterations < self.max_iterations:
            iterations += 1

            # step 1-2
            y_output = self.get_y()
            # print("y_output", y_output)

            # step 3
            deltas = []
            if not self.__expected_values:
                raise ValueError("expected_values is empty")
            deltas.append([[expected - y] for expected, y in zip(self.__expected_values, y_output)])
            # print("3. deltas", deltas)

            accuracy = max(abs(*d) for d in deltas[0])
            # print(accuracy)

            # step 4
            for i in range(self.__number_of_hidden_layers, 0, -1):
                temp_deltas = []
                for j in range(self.sizes_of_hidden_layers[i - 1]):
                    temp_deltas.append([
                        self.activation_func(self.__s_of_layers[i][j]) * sum(d) / len(d) * w
                        for d, w in zip(deltas[-1], self.weights[i][j])
                    ])
                deltas.append(temp_deltas)
            # print("4. deltas", deltas)

            # step 6
            w_deltas = []

            # for hidden and output layers
            for i in range(self.__number_of_hidden_layers, 0, -1):
                w_delta = []
                for j in range(self.sizes_of_hidden_layers[i - 1]):
                    w_delta.append([self.learning_rate * d
                                    for d in deltas[self.__number_of_hidden_layers - i + 1][j]])
                w_deltas.insert(0, w_delta)

            # for input layer
            w_deltas_for_input_layer = []
            for i, weights in enumerate(self.weights[0]):
                w_delta = []
                for j, weight in enumerate(weights):
                    w_delta.append(
                        weight * self.__s_of_layers[0][i] * self.learning_rate *
                        sum(deltas[-1][j]) / len(deltas[-1][j])
                    )
                w_deltas_for_input_layer.append(w_delta)
            w_deltas.insert(0, w_deltas_for_input_layer)
            # print("6. w_deltas", w_deltas)

            # step 7
            for i, weights_list in enumerate(w_deltas):
                for j, weights in enumerate(weights_list):
                    for k, weight in enumerate(weights):
                        # print(weight, "+", w_deltas[i][j][k])
                        self.weights[i][j][k] += w_deltas[i][j][k]
                # print(self.weights[i])

            # print("7. self.weights", self.weights)
            # print(self.__s_of_layers[0])
            # print()

        return iterations

    def get_y(self) -> list[float]:
        if not self.__s_of_layers:
            raise ValueError("input data is empty")

        # step 1
        for i in range(self.__number_of_hidden_layers):
            weighted_sums = []
            for j in range(self.sizes_of_hidden_layers[i]):
                weighted_sums.append(sum(x * w for x, w in zip(
                    self.__s_of_layers[i], list(map(itemgetter(j), self.weights[i])))))
            self.__s_of_layers[i + 1] = weighted_sums  # __s_of_layers[0] == input_data

        # step 2
        y_output = []
        for i in range(self.size_of_output_layer):
            y_output.append(sum(s * w for s, w in zip(
                self.__s_of_layers[-1], list(map(itemgetter(i), self.weights[-1])))))

        # print("s", self.__s_of_layers)

        return y_output
