import matplotlib.pyplot as plt

import random
import math


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

        # Селекція
        selected_parents = selection(population, fitness)

        # Схрещування
        offspring = crossover(selected_parents)

        # Мутація
        mutated_offspring = mutation(offspring)

        # Заміна популяції новими індивідами
        population = mutated_offspring

    return best_solution, best_fitness, worst_solution, worst_fitness, fitness_history


def selection(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    selected_parents = random.choices(population, weights=probabilities, k=len(population))
    return selected_parents


def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        child = (parent1 + parent2) / 2.0
        offspring.append(child)
    return offspring


def mutation(offspring):
    mutated_offspring = []
    mutation_rate = 0.1
    for child in offspring:
        if random.random() < mutation_rate:
            mutated_child = child + random.uniform(-0.5, 0.5)
            mutated_offspring.append(mutated_child)
        else:
            mutated_offspring.append(child)
    return mutated_offspring


def evaluate_fitness(individual):
    x = individual
    fitness = 5 * math.sin(x) * math.cos((x**2) + (1 / x))**2
    return fitness


if __name__ == "__main__":

    best_solution, best_fitness, worst_solution, worst_fitness, fitness_history = genetic_algorithm()

    print("Найкраще рішення (максимум):", best_solution)
    print("Найкращий фітнес (максимум):", best_fitness)
    print("Найгірше рішення (мінімум):", worst_solution)
    print("Найгірший фітнес (мінімум):", worst_fitness)

    x_values = [x / 100 for x in range(100 * 1, 100 * 11)]
    y_values = [evaluate_fitness(x) for x in x_values]

    plt.plot(x_values, y_values)
    plt.xlabel('x')
    plt.ylabel('Y(x)')
    plt.title('Графік функції Y(x) = 5 * sin(x) * cos((x**2) + (1 / x))**2')
    plt.grid(True)
    plt.show()
