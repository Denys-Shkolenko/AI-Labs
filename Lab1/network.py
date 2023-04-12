from typing import Sequence

from functions import derivative_err_func
from neuron import Neuron


class Network:

    def __init__(self, input_layer: Sequence[Neuron],
                 init_weights_for_hidden_layer: Sequence[Sequence[float]],
                 init_weights_for_output_neuron: Sequence[float],
                 expected_value: float,
                 hidden_layer_quantity=3,
                 learning_rate=0.1):
        self.hidden_layer_quantity = hidden_layer_quantity
        self.learning_rate = learning_rate

        self.input_layer = input_layer
        self.weights_for_hidden_layer = init_weights_for_hidden_layer
        self.weights_for_output_neuron = init_weights_for_output_neuron
        self.expected_value = expected_value

    @property
    def input_layer(self):
        return self.__input_layer

    @input_layer.setter
    def input_layer(self, input_layer: Sequence[Neuron]):
        if not len(input_layer):
            raise ValueError("input_layer cannot be empty")
        if not all(isinstance(neuron, Neuron) for neuron in input_layer):
            raise TypeError("each item of the input_layer must be a Neuron")
        self.__input_layer = input_layer

    @property
    def weights_for_hidden_layer(self):
        return self.__weights_for_hidden_layer

    @weights_for_hidden_layer.setter
    def weights_for_hidden_layer(self, weights_for_hidden_layer):
        if len(weights_for_hidden_layer) != len(self.input_layer):
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
    def weights_for_output_neuron(self, weights_for_output_neuron):
        if len(weights_for_output_neuron) != self.hidden_layer_quantity:
            raise ValueError("there must be weights for each neuron from "
                             "the hidden layer")
        for weight in weights_for_output_neuron:
            if not isinstance(weight, float):
                raise TypeError("each weight must be a float")
        self.__weights_for_output_neuron = list(weights_for_output_neuron)

    @property
    def expected_value(self):
        return self.__expected_value

    @expected_value.setter
    def expected_value(self, expected_value):
        if not isinstance(expected_value, float):
            raise TypeError("expected_value must be a float")
        self.__expected_value = expected_value

    @property
    def hidden_layer_quantity(self):
        return self.__hidden_layer_quantity

    @hidden_layer_quantity.setter
    def hidden_layer_quantity(self, hidden_layer_quantity):
        if not isinstance(hidden_layer_quantity, int):
            raise TypeError("hidden_layer_quantity must be an int")
        if hidden_layer_quantity < 1:
            raise ValueError("hidden_layer_quantity must be greater than 0")
        self.__hidden_layer_quantity = hidden_layer_quantity

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")
        if not 0.0 < learning_rate < 1.0:
            raise ValueError("learning_rate must be in range (0; 1)")
        self.__learning_rate = learning_rate

    def start_training(self):
        # step 1
        s_hidden_layer = []
        for i in range(self.hidden_layer_quantity):
            s_hidden_layer.append(sum(x * w for x, w in zip(
                self.input_layer, self.weights_for_hidden_layer[i])))
        print("1) s_hidden_layer =", s_hidden_layer)

        # step 2
        y_output = sum(s * w for s, w in zip(
                s_hidden_layer, self.weights_for_output_neuron))
        print("2) y_output =", y_output)

        # step 3
        delta = self.expected_value - y_output
        print("3) delta =", delta)

        # step 4
        delta_hidden_layer = []
        for i in range(self.hidden_layer_quantity):
            delta_hidden_layer.append(
                delta * self.weights_for_output_neuron[i] *
                derivative_err_func(s_hidden_layer[i]))
        print("4) delta_hidden_layer =", delta_hidden_layer)

        # step 6
        delta_w_output_layer = []
        for i in range(self.hidden_layer_quantity):
            delta_w_output_layer.append(
                delta_hidden_layer[i] * self.learning_rate)
        print("6) delta_w_output_layer =", delta_w_output_layer)

        delta_w_hidden_layer = []
        for sequence in self.weights_for_hidden_layer:
            delta_w_hidden_layer.append(
                [self.learning_rate * self.input_layer[i] *
                 delta_hidden_layer[i] * weight
                 for i, weight in enumerate(sequence)])
        print("6) delta_w_hidden_layer =", delta_w_hidden_layer)

        # step 7
        for i, weight in enumerate(self.weights_for_output_neuron):
            self.weights_for_output_neuron[i] = weight - delta_w_output_layer[i]
        print("7) weights_for_output_neuron =", self.weights_for_output_neuron)

        for i, sequence in enumerate(self.weights_for_hidden_layer):
            for j, weight in enumerate(sequence):
                self.weights_for_hidden_layer[i][j] = weight - delta_w_hidden_layer[i][j]

        print("7) weights_for_hidden_layer =", self.weights_for_hidden_layer)
