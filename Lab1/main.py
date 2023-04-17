from tabulate import tabulate

from network import Network

if __name__ == "__main__":

    # init_weights_for_hidden_layer = ((w14, w15, w16),
    #                                  (w24, w25, w26),
    #                                  (w34, w35, w36)) = ((0.2, 0.4, 0.2),
    #                                                      (0.2, 0.5, 2.4),
    #                                                      (0.3, 1.6, 0.5))
    #
    # init_weights_for_output_neuron = (w47, w57, w67) = (0.3, 0.5, 0.1)

    init_weights_for_hidden_layer = ((w14, w15, w16),
                                     (w24, w25, w26),
                                     (w34, w35, w36)) = ((1.0, 1.0, 1.0),
                                                         (1.0, 1.0, 1.0),
                                                         (1.0, 1.0, 1.0))

    init_weights_for_output_neuron = (w47, w57, w67) = (1.0, 1.0, 1.0)

    input_data = (2.57, 4.35, 1.27, 5.46, 1.30, 4.92, 1.31,
                  4.14, 1.97, 5.67, 0.92, 4.76, 1.72, 4.44, 1.49)

    network = Network(init_weights_for_hidden_layer,
                      init_weights_for_output_neuron,
                      max_iterations=1_000_000,
                      learning_rate=0.1)


    print("*" * 30, "TRAINING", "*" * 30)
    table = []
    for i in range(len(input_data) - 5):
        network.input_layer = (input_data[i],
                               input_data[i + 1],
                               input_data[i + 2])
        network.expected_value = input_data[i + 3]

        iteration = network.start_training()
        y = network.get_y()

        table.append([i + 1, ", ".join(str(num) for num in network.input_layer),
                      network.expected_value, round(y, 4), iteration])

    print(tabulate(table, headers=["No.", "Input Layer", "Expected Value",
                                   "y", "Iterations"], tablefmt="orgtbl"))


    print("\n\n" + "*" * 30, "TESTING", "*" * 30)
    table = []
    for i in range(len(input_data) - 3):
        network.input_layer = (input_data[i],
                               input_data[i + 1],
                               input_data[i + 2])
        network.expected_value = input_data[i + 3]

        y = network.get_y()
        expected = network.expected_value
        delta = abs(y - expected)

        table.append([i + 1, ", ".join(str(num) for num in network.input_layer),
                      network.expected_value, round(y, 4), round(delta, 4)])

    print(tabulate(table, headers=["No.", "Input Layer", "Expected Value",
                                   "y", "Delta"], tablefmt="orgtbl"))
