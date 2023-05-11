from tabulate import tabulate

from functions import *
from network import Network

if __name__ == "__main__":

    INIT_WEIGHTS_FOR_HIDDEN_LAYER = ((w14, w15, w16),
                                     (w24, w25, w26),
                                     (w34, w35, w36)) = ((2.0, 0.1, 1.1),
                                                         (0.1, 2.0, 0.9),
                                                         (2.0, 0.1, 1.0))

    INIT_WEIGHTS_FOR_OUTPUT_NEURON = (w47, w57, w67) = (1.1, 0.9, 1.0)

    # INIT_WEIGHTS_FOR_HIDDEN_LAYER = ((w14, w15, w16),
    #                                  (w24, w25, w26),
    #                                  (w34, w35, w36)) = ((1.0, 1.0, 1.0),
    #                                                      (1.0, 1.0, 1.0),
    #                                                      (1.0, 1.0, 1.0))
    #
    # INIT_WEIGHTS_FOR_OUTPUT_NEURON = (w47, w57, w67) = (1.0, 1.0, 1.0)

    INPUT_DATA = (2.57, 4.35, 1.27, 5.46, 1.30, 4.92, 1.31,
                  4.14, 1.97, 5.67, 0.92, 4.76, 1.72, 4.44, 1.49)

    network2 = Network(INIT_WEIGHTS_FOR_HIDDEN_LAYER,
                       INIT_WEIGHTS_FOR_OUTPUT_NEURON,
                       max_iterations=1_000_000,
                       learning_rate=0.1)

    for _ in range(1000):
        print("*" * 30, "TRAINING", "*" * 30)
        table2 = []
        network2.start_training2()
        for i in range(len(INPUT_DATA) - 3):
            network2.input_layer = (INPUT_DATA[i],
                                    INPUT_DATA[i + 1],
                                    INPUT_DATA[i + 2])
            network2.expected_value = INPUT_DATA[i + 3]

            y = network2.get_y()
            delta = abs(y - network2.expected_value)

            table2.append([i + 1, ", ".join(str(num) for num in network2.input_layer),
                          network2.expected_value, round(y, 4), round(delta, 4)])

        print(tabulate(table2, headers=["No.", "Input Layer", "Expected Value",
                                        "y", "Delta"], tablefmt="orgtbl"))



    # network = Network(INIT_WEIGHTS_FOR_HIDDEN_LAYER,
    #                   INIT_WEIGHTS_FOR_OUTPUT_NEURON,
    #                   max_iterations=1_000_000,
    #                   learning_rate=0.1)
    #
    # print("*" * 30, "TRAINING", "*" * 30)
    # table = []
    # for i in range(len(INPUT_DATA) - 5):
    #     network.input_layer = (INPUT_DATA[i],
    #                            INPUT_DATA[i + 1],
    #                            INPUT_DATA[i + 2])
    #     network.expected_value = INPUT_DATA[i + 3]
    #
    #     iteration = network.start_training()
    #     y = network.get_y()
    #     # print("h:", network.weights_for_hidden_layer)
    #     # print("o:", network.weights_for_output_neuron)
    #     # print("-")
    #
    #     table.append([i + 1, ", ".join(str(num) for num in network.input_layer),
    #                   network.expected_value, round(y, 4), iteration])
    #
    # print(tabulate(table, headers=["No.", "Input Layer", "Expected Value",
    #                                "y", "Iterations"], tablefmt="orgtbl"))
    #
    # print("\n\n" + "*" * 30, "TESTING", "*" * 30)
    # table = []
    # for i in range(len(INPUT_DATA) - 3):
    #     network.input_layer = (INPUT_DATA[i],
    #                            INPUT_DATA[i + 1],
    #                            INPUT_DATA[i + 2])
    #     network.expected_value = INPUT_DATA[i + 3]
    #
    #     y = network.get_y()
    #     expected = network.expected_value
    #     delta = abs(y - expected)
    #
    #     table.append([i + 1, ", ".join(str(num) for num in network.input_layer),
    #                   network.expected_value, round(y, 4), round(delta, 4)])
    #
    # print(tabulate(table, headers=["No.", "Input Layer", "Expected Value",
    #                                "y", "Delta"], tablefmt="orgtbl"))

    # logic_OR_input = ((0.0, 0.0, 0.0),
    #                   (1.0, 0.0, 1.0),
    #                   (0.0, 1.0, 1.0),
    #                   (1.0, 1.0, 1.0))
    #
    # logic_OR_network = Network(
    #     init_weights_for_hidden_layer=((1.0,), (1.0,)),
    #     init_weights_for_output_neuron=(1.0,),
    #     input_layer=(0.0, 0.0),
    #     input_layer_quantity=2,
    #     hidden_layer_quantity=1,
    #     activation_func=logic_OR_activation_func)
    #
    # table = []
    # for i, row in enumerate(logic_OR_input):
    #     logic_OR_network.input_layer = (row[0], row[1])
    #     logic_OR_network.expected_value = row[2]
    #
    #     iteration = logic_OR_network.start_training()
    #     y = logic_OR_network.get_y()
    #     # print(logic_OR_network.)
    #
    #     table.append([i + 1, ", ".join(str(num) for num in logic_OR_network.input_layer),
    #                   logic_OR_network.expected_value, round(y, 4), iteration])
    #
    # print("\n\n")
    # print(tabulate(table, headers=["No.", "Input Layer", "Expected Value",
    #                                "y", "Iterations"], tablefmt="orgtbl"))
