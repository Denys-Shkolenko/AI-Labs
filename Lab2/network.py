from operator import itemgetter

import numpy as np
from typing import Sequence

from functions import derivative_err_func


class Network:

    def __init__(self,
                 size_of_input_layer=36,
                 sizes_of_hidden_layers=(36, 30),
                 size_of_output_layer=2,
                 default_weight=1.0,
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
        self.__s_of_layers = []

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
        self.__s_of_layers.insert(0, input_data)

    def set_expected_values(self, expected_values: Sequence[float]):
        if not expected_values:
            raise ValueError("expected_values cannot be empty")
        if len(expected_values) != self.size_of_output_layer:
            raise ValueError("the len of expected_values must be equal to the "
                             "size_of_output_layer parameter")
        if not all(isinstance(number, float) for number in expected_values):
            raise TypeError("each item of the input_data must be an int")
        self.__expected_values = expected_values

    def start_training(self) -> int:
        iterations = 0
        # delta = self.eps + 1  # to start the loop
        while abs(100) > self.eps and iterations < self.max_iterations:
            iterations += 1

            # step 1-2
            y_output = self.get_y()

            # step 3
            deltas = []
            if not self.__expected_values:
                raise ValueError("expected_values is empty")
            deltas.append([expected - y for expected, y in zip(self.__expected_values, y_output)])

            # step 4
            for i in range(self.__number_of_hidden_layers, 0, -1):
                average_deltas = []
                for j in range(self.sizes_of_hidden_layers[i - 1]):
                    average_deltas.append(
                        sum(self.activation_func(self.__s_of_layers[i][j]) * d * w
                            for d, w in zip(deltas[-1], self.weights[i][j]))
                        / self.size_of_output_layer)
                deltas.append(average_deltas)

            # step 6
            delta_w_output_layer = []
            for i in range(self.hidden_layer_quantity):
                delta_w_output_layer.append(
                    delta_hidden_layer[i] * self.learning_rate)

            delta_w_hidden_layer = []
            for i, sequence in enumerate(self.weights_for_hidden_layer):
                delta_w_hidden_layer.append(
                    [self.learning_rate * self.input_layer[i] *
                     delta_hidden_layer[i] * weight
                     for weight in sequence])

            # step 7
            for i, weight in enumerate(self.weights_for_output_neuron):
                self.weights_for_output_neuron[i] = weight + delta_w_output_layer[i]

            for i, sequence in enumerate(self.weights_for_hidden_layer):
                for j, weight in enumerate(sequence):
                    self.weights_for_hidden_layer[i][j] = weight + delta_w_hidden_layer[i][j]

        return iterations

    def get_y(self) -> list[float]:
        if not self.__s_of_layers:
            raise ValueError("input data is empty")

        # step 1
        for i in range(self.number_of_hidden_layers):
            weighted_sums = []
            for j in range(self.sizes_of_hidden_layers[i]):
                weighted_sums.append(sum(x * w for x, w in zip(
                    self.__s_of_layers[i], list(map(itemgetter(j), self.weights[i])))))
            self.__s_of_layers.insert(i + 1, weighted_sums)  # __s_of_layers[0] == input_data

        # step 2
        y_output = []
        for i in range(self.size_of_output_layer):
            y_output.append(sum(s * w for s, w in zip(
                self.__s_of_layers[-1], list(map(itemgetter(i), self.weights[-1])))))

        return y_output
