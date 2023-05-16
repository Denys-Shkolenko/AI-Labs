from functions import *
from network import Network

if __name__ == "__main__":
    network = Network(size_of_input_layer=3,
                      sizes_of_hidden_layers=(4, 3),
                      size_of_output_layer=2,
                      max_iterations=10_000,
                      default_weight=1.0,
                      learning_rate=0.1,
                      eps=0.0001)
    INPUT_DATA = (1, 0, 1)
    EXPECTED_VALUES = (1.0, 0.0)
    network.set_input_data(INPUT_DATA)
    network.set_expected_values(EXPECTED_VALUES)

    network.start_training()



