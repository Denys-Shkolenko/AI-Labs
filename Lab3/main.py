import math
import random
import matplotlib.pyplot as plt

# Функція для обчислення фітнесу
def evaluate_fitness(x):
    return 5 * math.sin(x) * math.cos((x**2) + (1 / x))**2

# Генетичний алгоритм
def genetic_algorithm():
    population_size = 100
    generations = 50
    best_solution = None
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

    return best_solution, best_fitness, worst_fitness, fitness_history

# Запуск генетичного алгоритму
best_solution, best_fitness, worst_fitness, fitness_history = genetic_algorithm()

# Виведення результатів
# print("Найкраще рішення (максимум):", best_solution)
print("максимум:", best_fitness)
print("мінімум:", worst_fitness)

# Побудова графіка історії фітнесу
plt.plot(fitness_history)
plt.xlabel("Покоління")
plt.ylabel("Фітнес")
plt.show()
