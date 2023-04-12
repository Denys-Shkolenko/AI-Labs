from typing import Sequence

from neuron import Neuron


class Network:

    def __init__(self, input_layer: Sequence[Neuron], expected_value: float):
        self.input_layer = input_layer
        self.expected_value = expected_value

    @property
    def input_layer(self):
        return self.__layer_1

    @input_layer.setter
    def input_layer(self, layer_1: Sequence[Neuron]):
        if not all(isinstance(neuron, Neuron) for neuron in layer_1):
            raise TypeError("each item of the input_layer must be a Neuron")
        self.__layer_1 = layer_1

    @property
    def expected_value(self):
        return self.__expected_value

    @expected_value.setter
    def expected_value(self, expected_value):
        if not isinstance(expected_value, float):
            raise TypeError("expected_value must be a float")
        self.__expected_value = expected_value
