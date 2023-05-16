from functions import *
from network import Network

if __name__ == "__main__":
    network = Network(size_of_input_layer=3,
                      sizes_of_hidden_layers=(3, 2),
                      size_of_output_layer=2)
    INPUT_DATA = (1, 1, 1)
    EXPECTED_VALUES = (10.0, 10.0)
    network.set_input_data(INPUT_DATA)
    network.set_expected_values(EXPECTED_VALUES)

    network.start_training()

