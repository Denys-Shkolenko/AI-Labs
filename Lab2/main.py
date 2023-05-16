import json

from tabulate import tabulate

from functions import *
from network import Network

if __name__ == "__main__":
    network = Network(size_of_input_layer=36,
                      sizes_of_hidden_layers=(6,),
                      size_of_output_layer=3,
                      default_weight=0.1,
                      learning_rate=0.1,
                      eps=0.001,
                      max_iterations=1_000,
                      activation_func=derivative_err_func)

    with open("training_data.json") as training_data:
        training_data = json.load(training_data)

    # training
    table = []
    print("*" * 20, "TRAINING", "*" * 20)
    for data in training_data:
        network.set_input_data(data["symbol"])
        network.set_expected_values(data["8-bit code"])
        iterations = network.start_training()

        table.append([
            data["answer"],
            ''.join(str(num) for num in data["8-bit code"]),
            '   '.join(str(round(num, 8)) for num in network.get_y()),
            iterations
        ])

    print(tabulate(table, headers=["Image", "Expected Value", "y", "Iterations"], tablefmt="orgtbl"))
    print("\n")

    # testing
    table = []
    print("*" * 20, "TESTING", "*" * 20)
    for data in training_data:
        network.set_input_data(data["symbol"])
        network.set_expected_values(data["8-bit code"])

        table.append([
            data["answer"],
            ''.join(str(num) for num in data["8-bit code"]),
            '   '.join(str(round(num, 8)) for num in network.get_y())
        ])

    print(tabulate(table, headers=["Image", "Expected Value", "y", "Iterations"], tablefmt="orgtbl"))