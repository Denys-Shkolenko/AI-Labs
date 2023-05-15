from tabulate import tabulate

from functions import *
from network import Network

if __name__ == "__main__":

    # INIT_WEIGHTS_FOR_HIDDEN_LAYER = ((w14, w15, w16),
    #                                  (w24, w25, w26),
    #                                  (w34, w35, w36)) = ((1.1, 0.2, 0.1),
    #                                                      (0.1, 2.3, 0.2),
    #                                                      (1.0, 0.1, 0.1))
    #
    # INIT_WEIGHTS_FOR_OUTPUT_NEURON = (w47, w57, w67) = (0.8, 0.7, 0.9)

    INIT_WEIGHTS_FOR_HIDDEN_LAYER = ((w14, w15, w16),
                                     (w24, w25, w26),
                                     (w34, w35, w36)) = ((1.0, 1.0, 1.0),
                                                         (1.0, 1.0, 1.0),
                                                         (1.0, 1.0, 1.0))

    INIT_WEIGHTS_FOR_OUTPUT_NEURON = (w47, w57, w67) = (1.0, 1.0, 1.0)

    INPUT_DATA = (2.57, 4.35, 1.27, 5.46, 1.30, 4.92, 1.31,
                  4.14, 1.97, 5.67, 0.92, 4.76, 1.72, 4.44, 1.49)

    # training with average weights

    network1 = Network(INIT_WEIGHTS_FOR_HIDDEN_LAYER,
                       INIT_WEIGHTS_FOR_OUTPUT_NEURON,
                       max_iterations=1_000_000,
                       learning_rate=0.1,
                       input_data=INPUT_DATA)

    print("1.")
    print("*" * 20, "TRAINING WITH AVERAGE WEIGHTS", "*" * 20)
    table1 = []
    network1.start_training_with_average_weights(1000)
    for i in range(len(INPUT_DATA) - 3):
        network1.input_layer = (INPUT_DATA[i],
                                INPUT_DATA[i + 1],
                                INPUT_DATA[i + 2])
        network1.expected_value = INPUT_DATA[i + 3]

        y = network1.get_y()
        delta = abs(y - network1.expected_value)

        table1.append([i + 1, ", ".join(str(num) for num in network1.input_layer),
                      network1.expected_value, round(y, 4), round(delta, 4)])

    print(tabulate(table1, headers=["No.", "Input Layer", "Expected Value",
                                    "y", "Delta"], tablefmt="orgtbl"))
    print("\n")

    # training

    network2 = Network(INIT_WEIGHTS_FOR_HIDDEN_LAYER,
                       INIT_WEIGHTS_FOR_OUTPUT_NEURON,
                       max_iterations=1_000_000,
                       learning_rate=0.1)

    print("2.")
    print("*" * 30, "TRAINING", "*" * 30)
    table2 = []
    for i in range(len(INPUT_DATA) - 5):
        network2.input_layer = (INPUT_DATA[i],
                                INPUT_DATA[i + 1],
                                INPUT_DATA[i + 2])
        network2.expected_value = INPUT_DATA[i + 3]

        iteration = network2.start_training()
        y = network2.get_y()

        table2.append([i + 1, ", ".join(str(num) for num in network2.input_layer),
                      network2.expected_value, round(y, 4), iteration])

    print(tabulate(table2, headers=["No.", "Input Layer", "Expected Value",
                                    "y", "Iterations"], tablefmt="orgtbl"))
    print("\n")

    # testing

    print("*" * 30, "TESTING", "*" * 30)
    table3 = []
    for i in range(len(INPUT_DATA) - 3):
        network2.input_layer = (INPUT_DATA[i],
                                INPUT_DATA[i + 1],
                                INPUT_DATA[i + 2])
        network2.expected_value = INPUT_DATA[i + 3]

        y = network2.get_y()
        expected = network2.expected_value
        delta = abs(y - expected)

        table3.append([i + 1, ", ".join(str(num) for num in network2.input_layer),
                      network2.expected_value, round(y, 4), round(delta, 4)])

    print(tabulate(table3, headers=["No.", "Input Layer", "Expected Value",
                                    "y", "Delta"], tablefmt="orgtbl"))
    print("\n")

    # logic functions

    network_for_logic_functions = Network(
        init_weights_for_hidden_layer=((1.0,), (1.0,)),
        init_weights_for_output_neuron=(1.0,),
        input_layer=(0.0, 0.0),
        input_layer_quantity=2,
        hidden_layer_quantity=1)

    logic_OR_input = ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 1.0),
                      (0.0, 1.0, 1.0),
                      (1.0, 1.0, 1.0))

    logic_AND_input = ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0),
                       (0.0, 1.0, 0.0),
                       (1.0, 1.0, 1.0))

    logic_XOR_input = ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 1.0),
                       (0.0, 1.0, 1.0),
                       (1.0, 1.0, 0.0))

    logic_functions = {
        logic_OR_input: [logic_OR_activation_func, "LOGIC OR"],
        logic_AND_input: [logic_AND_activation_func, "LOGIC AND"],
        logic_XOR_input: [logic_XOR_activation_func, "LOGIC XOR"]
    }
    for input_data, param in logic_functions.items():
        print("*" * 20, param[1], "*" * 20)
        table4 = []
        for i, row in enumerate(input_data):
            network_for_logic_functions.input_layer = (row[0], row[1])
            network_for_logic_functions.expected_value = row[2]

            y = param[0](network_for_logic_functions.get_y())

            table4.append([i + 1, ", ".join(str(num) for num in network_for_logic_functions.input_layer),
                          network_for_logic_functions.expected_value, round(y, 4)])

        print(tabulate(table4, headers=["No.", "Input Layer", "Expected Value",
                                        "y"], tablefmt="orgtbl"))
        print("\n")

    logic_NOT_network = Network(
        init_weights_for_hidden_layer=((1.0,),),
        init_weights_for_output_neuron=(-1.5,),
        input_layer=(0.0,),
        input_layer_quantity=1,
        hidden_layer_quantity=1)

    logic_NOT_input = ((0.0, 1.0),
                       (1.0, 0.0))

    print("*" * 20, "LOGIC NOT", "*" * 20)
    table5 = []
    for i, row in enumerate(logic_NOT_input):
        logic_NOT_network.input_layer = (row[0],)
        logic_NOT_network.expected_value = row[1]

        y = logic_NOT_activation_func(logic_NOT_network.get_y())

        table5.append([i + 1, ", ".join(str(num) for num in logic_NOT_network.input_layer),
                      logic_NOT_network.expected_value, round(y, 4)])

    print(tabulate(table5, headers=["No.", "Input Layer", "Expected Value",
                                    "y"], tablefmt="orgtbl"))
    print("\n")
