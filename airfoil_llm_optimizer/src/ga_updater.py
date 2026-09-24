import logging
import numpy as np

logger = logging.getLogger(__name__)


def ga_update(input_params, num_children=None):
    """
    Genetic Algorithm (GA) core update function. Generates multiple offspring
    through elitism, crossover and mutation, then returns the population mean
    as the sampling center for the next generation.

    Args:
        input_params (list): [population, fitness_scores, bounds, gbest_fitness, gbest]
            - population: list of individual dicts with 'A_u' and 'A_l' keys
            - fitness_scores: fitness values matching population size
            - bounds: dict with 'A_u': (min, max), 'A_l': (min, max)
            - gbest_fitness: global best fitness (float)
            - gbest: global best individual dict
        num_children (int): number of children to generate (default: population size)

    Returns:
        new_mean (dict): child population mean {'A_u': [...], 'A_l': [...]}
    """
    population, fitness_scores, bounds, gbest_fitness, gbest = input_params
    pop_size = len(population)
    if num_children is None:
        num_children = pop_size

    # ----- Step 1: GA hyperparameters -----
    cross_prob = 0.8       # crossover probability
    mutate_prob = 0.1      # mutation probability
    elite_rate = 0.2       # elite preservation ratio

    # ----- Step 2: Fitness preprocessing for roulette wheel selection -----
    fitness_arr = np.array(fitness_scores)
    fitness_min = np.min(fitness_arr)
    fitness_non_neg = fitness_arr - fitness_min + 1e-6
    fitness_sum = np.sum(fitness_non_neg)
    select_prob = fitness_non_neg / fitness_sum

    # ----- Step 3: Build elite pool -----
    elite_num = max(1, int(pop_size * elite_rate))
    elite_idx = np.argsort(fitness_arr)[-elite_num:]
    elite_individuals = [population[i] for i in elite_idx]
    # gbest 为空时用种群内最优个体兜底
    if gbest is None:
        gbest = population[int(np.argmax(fitness_arr))]
    # Ensure gbest is in elite pool
    if not any(np.allclose(ind['A_u'], gbest['A_u']) and np.allclose(ind['A_l'], gbest['A_l'])
               for ind in elite_individuals):
        elite_individuals.append(gbest)

    # ----- Step 4: Bounds and mutation step sizes -----
    au_min, au_max = bounds['A_u']
    al_min, al_max = bounds['A_l']
    au_mutate_step = (au_max - au_min) * 0.1
    al_mutate_step = (al_max - al_min) * 0.1

    # ----- Step 5: Generate num_children offspring -----
    children_Au = []
    children_Al = []

    for _ in range(num_children):
        # Selection: elite parent 1 + roulette-selected parent 2
        parent1 = elite_individuals[np.random.randint(len(elite_individuals))]
        parent2_idx = np.random.choice(pop_size, p=select_prob)
        parent2 = population[parent2_idx]

        p1_Au = np.array(parent1['A_u'])
        p1_Al = np.array(parent1['A_l'])
        p2_Au = np.array(parent2['A_u'])
        p2_Al = np.array(parent2['A_l'])

        # Crossover (arithmetic crossover)
        if np.random.rand() < cross_prob:
            cross_weight = np.random.rand(len(p1_Au))
            child_Au = cross_weight * p1_Au + (1 - cross_weight) * p2_Au
            child_Al = cross_weight * p1_Al + (1 - cross_weight) * p2_Al
        else:
            child_Au = p1_Au.copy()
            child_Al = p1_Al.copy()

        # Mutation (Gaussian mutation)
        mutate_mask_au = np.random.rand(len(child_Au)) < mutate_prob
        child_Au[mutate_mask_au] += np.random.normal(0, au_mutate_step, size=np.sum(mutate_mask_au))
        mutate_mask_al = np.random.rand(len(child_Al)) < mutate_prob
        child_Al[mutate_mask_al] += np.random.normal(0, al_mutate_step, size=np.sum(mutate_mask_al))

        # Boundary clipping
        child_Au = np.clip(child_Au, au_min, au_max)
        child_Al = np.clip(child_Al, al_min, al_max)

        children_Au.append(child_Au)
        children_Al.append(child_Al)

    # ----- Step 6: Compute mean of all children as next-gen sampling center -----
    new_mean = {
        'A_u': np.mean(children_Au, axis=0).tolist(),
        'A_l': np.mean(children_Al, axis=0).tolist(),
    }
    return new_mean
