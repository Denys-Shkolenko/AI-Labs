from operator import itemgetter

import numpy as np
from typing import Sequence

from functions import derivative_err_func


class Network:

    def __init__(self,
                 size_of_input_layer=36,
                 sizes_of_hidden_layers=(36, 30),
                 size_of_output_layer=2,
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

        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iterations = max_iterations
        self.activation_func = activation_func

        # save the state of the weighted sums used in the get_y() and
        # start_training() functions
        self.s_hidden_layer = []

    @property
    def weights_for_hidden_layer(self):
        return self.__weights_for_hidden_layer

    @weights_for_hidden_layer.setter
    def weights_for_hidden_layer(self, weights_for_hidden_layer: Sequence[Sequence[float]]):
        if len(weights_for_hidden_layer) != self.input_layer_quantity:
            raise ValueError("there must be weights for each neuron from "
                             "the input layer")
        for sequence in weights_for_hidden_layer:
            if len(sequence) != self.hidden_layer_quantity:
                raise ValueError("there must be weights for each neuron from "
                                 "the hidden layer")
            for weight in sequence:
                if not isinstance(weight, float):
                    raise TypeError("each weight must be a float")
        self.__weights_for_hidden_layer = list(
            list(item) for item in weights_for_hidden_layer)

    @property
    def weights_for_output_neuron(self):
        return self.__weights_for_output_neuron

    @weights_for_output_neuron.setter
    def weights_for_output_neuron(self, weights_for_output_neuron: Sequence[float]):
        if len(weights_for_output_neuron) != self.hidden_layer_quantity:
            raise ValueError("there must be weights for each neuron from "
                             "the hidden layer")
        for weight in weights_for_output_neuron:
            if not isinstance(weight, float):
                raise TypeError("each weight must be a float")
        self.__weights_for_output_neuron = list(weights_for_output_neuron)

    @property
    def input_layer(self):
        return self.__input_layer

    @input_layer.setter
    def input_layer(self, input_layer: Sequence[float]):
        if not len(input_layer):
            raise ValueError("input_layer cannot be empty")
        if len(input_layer) != self.input_layer_quantity:
            raise ValueError("the len of input_layer must be equal to the "
                             "input_layer_quantity parameter")
        if not all(isinstance(number, float) for number in input_layer):
            raise TypeError("each item of the input_layer must be a float")
        self.__input_layer = input_layer

    @property
    def expected_value(self):
        return self.__expected_value

    @expected_value.setter
    def expected_value(self, expected_value: float):
        if not isinstance(expected_value, float):
            raise TypeError("expected_value must be a float")
        self.__expected_value = expected_value

    @property
    def input_layer_quantity(self):
        return self.__input_layer_quantity

    @property
    def hidden_layer_quantity(self):
        return self.__hidden_layer_quantity

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

    def start_training(self) -> int:
        """
        Start the process of training the neural network until the output value
        is equal to the expected value with the specified accuracy (eps) or
        the maximum number of iterations is completed.

        Returns:
            int: number of iterations completed.
        """

        iterations = 0
        delta = self.eps + 1  # to start the loop
        while abs(delta) > self.eps and iterations < self.max_iterations:
            iterations += 1

            # step 1-2
            y_output = self.get_y()

            # step 3
            delta = self.expected_value - y_output

            # step 4
            delta_hidden_layer = []
            for i in range(self.hidden_layer_quantity):
                delta_hidden_layer.append(
                    delta * self.weights_for_output_neuron[i] *
                    self.activation_func(self.s_hidden_layer[i]))

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

    def get_y(self) -> float:
        """
        Get the value of the output neuron and write the weighted sums.

        Returns:
            float: the value of the output neuron
        """

        # step 1
        self.s_hidden_layer.clear()
        for i in range(self.hidden_layer_quantity):
            self.s_hidden_layer.append(sum(x * w for x, w in zip(
                self.input_layer, list(map(itemgetter(0), self.weights_for_hidden_layer)))))

        # step 2
        y_output = sum(s * w for s, w in zip(
            self.s_hidden_layer, self.weights_for_output_neuron))

        return y_output
