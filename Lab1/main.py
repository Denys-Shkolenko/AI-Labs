from neuron import Neuron
from network import Network


if __name__ == "__main__":

    init_weights_for_hidden_layer = ((w14, w15, w16),
                                     (w24, w25, w26),
                                     (w34, w35, w36)) = ((1.0, 1.0, 1.0),
                                                         (1.0, 1.0, 1.0),
                                                         (1.0, 1.0, 1.0))

    init_weights_for_output_neuron = (w47, w57, w67) = (1.0, 1.0, 1.0)

    input_layer = (Neuron(2.57), Neuron(4.35), Neuron(1.27))

    network = Network(input_layer,
                      init_weights_for_hidden_layer,
                      init_weights_for_output_neuron,
                      expected_value=5.46,
                      hidden_layer_quantity=3,
                      learning_rate=0.1)

    network.start_training()
