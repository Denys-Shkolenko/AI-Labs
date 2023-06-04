import matplotlib.pyplot as plt
import random
import math
from tabulate import tabulate


def fitness_function(x):
    return 5 * math.sin(x) * math.cos(x ** 2 + 1 / x) ** 2


def decode_chromosome(chromosome, xmin, xmax, nb):
    n = int("".join(map(str, chromosome)), 2)
    return xmin + n * (xmax - xmin) / (2 ** nb - 1)


def evaluate_population_fitness(population, xmin, xmax, nb):
    fitness_values = []
    for chromosome in population:
        x = decode_chromosome(chromosome, xmin, xmax, nb)
        fitness = fitness_function(x)
        fitness_values.append(fitness)
    return fitness_values


def create_initial_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(chromosome)
    return population


def genetic_algorithm(population_size, chromosome_length, xmin, xmax, nb, generations):
    population = create_initial_population(population_size, chromosome_length)

    table = []
    headers = ["Generation", "Best Solution (x, y)", "Worst Solution (x, y)"]

    for generation in range(generations):
        fitness_values = evaluate_population_fitness(population, xmin, xmax, nb)

        best_fitness = max(fitness_values)
        worst_fitness = min(fitness_values)

        best_chromosome = population[fitness_values.index(best_fitness)]
        worst_chromosome = population[fitness_values.index(worst_fitness)]

        x_best = decode_chromosome(best_chromosome, xmin, xmax, nb)
        y_best = best_fitness
        x_worst = decode_chromosome(worst_chromosome, xmin, xmax, nb)
        y_worst = worst_fitness

        table.append([generation + 1,
                      (round(x_best, 8), round(y_best, 8)),
                      (round(x_worst, 8), round(y_worst, 8))])

        new_population = [best_chromosome, worst_chromosome]

        # Generation of new chromosomes by crossover
        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            crossover_point = random.randint(1, chromosome_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            new_population.append(child1)
            new_population.append(child2)

        # Replacing the previous population with a new one
        population = new_population

    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    population_size = 50
    chromosome_length = 32
    xmin = 1
    xmax = 10
    nb = chromosome_length
    generations = 50

    genetic_algorithm(population_size, chromosome_length, xmin, xmax, nb, generations)

    x_values = [x / 100 for x in range(100 * 1, 100 * 11)]
    y_values = [fitness_function(x) for x in x_values]

    plt.plot(x_values, y_values)
    plt.xlabel("x")
    plt.ylabel("Y(x)")
    plt.title("Graph of the function Y(x) = 5sin(x)cos(x^2 + 1/x)^2")
    plt.grid(True)
    plt.show()
