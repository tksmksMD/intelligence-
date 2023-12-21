import random

def eval_func(chromosome):
    x, y, z = chromosome
    return 1 / (1 + (x - 2)**2 + (y + 1)**2 + (z - 1)**2),

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_probability):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_probability:
            # вибір одного гена для мутації 
            mutated_individual[i] += random.uniform(-0.5, 0.5)
    return tuple(mutated_individual)

def tournament_selection(population, tournament_size):
    # 
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda ind: eval_func(ind), reverse=True)
    return tournament[0]

def genetic_algorithm(population_size, generations, crossover_probability, mutation_probability, tournament_size):
    population = [(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(population_size)]

    for generation in range(generations):
        new_population = []
        for i in range(0, population_size, 2):
            # Вибір батьків за допомогою турнірного відбору
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            # Схрещення з певною ймовірністю
            if random.random() < crossover_probability:
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([mutate(child1, mutation_probability), mutate(child2, mutation_probability)])
            else:
                new_population.extend([mutate(parent1, mutation_probability), mutate(parent2, mutation_probability)])

        # Сортування нової популяції
        new_population.sort(key=lambda ind: eval_func(ind), reverse=True)

        # Відбір найкращих особин
        population = new_population[:population_size]

    # Визначення найкращого рішення
    best_solution = max(population, key=lambda ind: eval_func(ind))
    return eval_func(best_solution), best_solution

# Параметри генетичного алгоритму
population_size = 50
generations = 100
crossover_probability = 0.7
mutation_probability = 0.1
tournament_size = 5  #розмір турніру

# Запуск генетичного алгоритму
best_fitness, best_solution = genetic_algorithm(population_size, generations, crossover_probability, mutation_probability, tournament_size)

# Виведення результатів
print(f"Best Fitness: {best_fitness[0]}")
print(f"Best Solution: {best_solution}")
