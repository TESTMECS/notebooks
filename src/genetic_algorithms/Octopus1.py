import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    🐙 Octopus Search Algorithm (OSA)
    Overview

    The Octopus Search Algorithm (OSA) is a nature-inspired optimization algorithm that simulates the intelligent foraging behavior and remarkable camouflage abilities of octopuses to explore and exploit search spaces for finding optimal solutions. It models the population of octopuses continuously searching for the best camouflage, which represents a potential solution to an optimization problem.

    Key concepts include:

        Population of Octopuses: Each octopus represents a candidate solution in the search space. Their positions are updated based on their interactions with their environment and other octopuses.
        Camouflage Matrix: A matrix that stores the positions and fitness values of promising camouflages (solutions) discovered during the search.
        Color Change Model: Simulates an octopus's ability to match its color to its surroundings, influencing the octopus's movement towards potentially better areas in the search space.
        Shape Change Model: Mimics an octopus's ability to alter its body shape to blend in or mimic objects, allowing for exploration of the search space by referencing known good solutions.
        Parameters: Control the algorithm's behavior, such as the rate of color and shape changes, the probability of using each model, and the number of octopuses.

    Uses

    The Octopus Search Algorithm is a metaheuristic algorithm that can be applied to a wide range of optimization problems, including:

        Global Optimization: Finding the best possible solution in a complex search space.
        Parameter Tuning: Optimizing the parameters of machine learning models or other systems.
        Feature Selection: Identifying the most relevant features in a dataset.
        Engineering Design: Finding optimal designs for various engineering problems.
        Resource Allocation: Optimizing the distribution of resources.

    """
    )
    return


@app.cell
def _():


    import numpy as np
    import scipy

    def fitness(x):
      # Added return statement to return the calculated value
      return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

    # === params ======
    num_octopuses = 100
    num_dims = 2
    max_iter = 1000
    alpha = 0.01 # Color change rate factor.
    beta = 0.1 # Shape change rate factor.
    p = 0.5 # probability between color change and shape change.
    lower_bound = -10
    upper_bound = 10
    # ==================
    positions = np.random.uniform(lower_bound, upper_bound, (num_octopuses, num_dims))
    fitness_values = np.array([fitness(pos) for pos in positions])
    best_position = positions[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)

    # Initalize the camouflage positions and fitness values
    camoflages = np.tile(best_position, (num_octopuses, 1))
    camoflage_fitness = np.tile(best_fitness, num_octopuses)
    # === main loop ======
    for i in range(max_iter):
      # Loop over each octopus
      for j in range(num_octopuses):
        # Choose a random camouflage index.
        camouflage_index = np.random.randint(num_octopuses)
        # Calculate the distance between the octopus and the camouflage.
        distance = np.linalg.norm(positions[j] - camoflages[camouflage_index])
        # Choose either kcolor change model or shape change model based on a random parameter q between 0 and 1
        q = np.random.rand()

        if q < p:
          # Use color change model to update position.
          new_position = positions[j] + alpha * scipy.stats.levy_stable.rvs(1.5,0,size=num_dims) * (camoflages[camouflage_index] - positions[j])
        else:
          # Use shape change model to update position.
          new_position = camoflages[camouflage_index] + beta * distance * np.random.uniform(-1,1,size=num_dims)
        # Apply boundary conditions
        new_position = np.clip(new_position, lower_bound, upper_bound)
        # Evalute the fitness value of the new position.
        new_fitness = fitness(new_position)
        # Update the camouflage position and fitness value if the new fitness value is better.
        if new_fitness < camoflage_fitness[j]:
          camoflages[j] = new_position.copy()
          camoflage_fitness[j] = new_fitness
          if new_fitness < best_fitness:
            best_position = new_position.copy()
            best_fitness = new_fitness

        # Sort the octopuses by their fitness values in ascending order.
        sorted_indices = np.argsort(fitness_values)
        positions = positions[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        # Update the number of camoflages positions and fitness values by copying from the best octopuses.
        num_camoflages = num_octopuses - i * (num_octopuses - 1) // max_iter
        camoflages[:num_camoflages] = positions[:num_camoflages].copy()
        camoflage_fitness[:num_camoflages] = fitness_values[:num_camoflages].copy()

    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
    return


if __name__ == "__main__":
    app.run()
