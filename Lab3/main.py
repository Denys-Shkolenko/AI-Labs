import math
import random
import matplotlib.pyplot as plt


def evaluate_fitness(x):
    return 5 * math.sin(x) * math.cos((x**2) + (1 / x))**2


def genetic_algorithm():
    population_size = 100
    generations = 50
    best_solution = None
    worst_solution = None
    best_fitness = float('-inf')
    worst_fitness = float('inf')
    fitness_history = []

    for _ in range(generations):
        population = [random.uniform(1, 10) for _ in range(population_size)]

        fitness = [evaluate_fitness(x) for x in population]
        fitness_history.append(max(fitness))

        if max(fitness) > best_fitness:
            best_fitness = max(fitness)
            best_solution = population[fitness.index(max(fitness))]

        if min(fitness) < worst_fitness:
            worst_fitness = min(fitness)
            worst_solution = population[fitness.index(min(fitness))]

    return best_solution, best_fitness, worst_solution, worst_fitness, fitness_history


if __name__ == "__main__":

    best_solution, best_fitness, worst_solution, worst_fitness, fitness_history = genetic_algorithm()

    print("Найкраще рішення (максимум):", best_solution)
    print("Найкращий фітнес (максимум):", best_fitness)
    print("Найгірше рішення (мінімум):", worst_solution)
    print("Найгірший фітнес (мінімум):", worst_fitness)

    # plt.plot(fitness_history)
    # plt.xlabel("Покоління")
    # plt.ylabel("Фітнес")
    # plt.show()

    x_values = [x / 100 for x in range(100 * 1, 100 * 11)]
    y_values = [evaluate_fitness(x) for x in x_values]

    plt.plot(x_values, y_values)
    plt.xlabel('x')
    plt.ylabel('Y(x)')
    plt.title('Графік функції Y(x) = 5 * sin(x) * cos((x**2) + (1 / x))**2')
    plt.grid(True)
    plt.show()
