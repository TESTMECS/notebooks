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
    Honey Badger Algorithm (HBA) 🦡
    Overview 📊

    The Honey Badger Algorithm (HBA) is a nature-inspired optimization algorithm that mimics the foraging behavior of honey badgers. Key concepts include:

        Population of Honey Badgers: Each badger represents a potential solution to the optimization problem.
        Food Source Matrix: Stores the best solutions found so far, representing attractive foraging areas.
        Digging Model: Simulates the badger's movement towards promising food sources.
        Hunting Model: Simulates the badger's exploration around food sources to find better solutions.
        Control Parameters: Factors like digging depth, hunting radius, and switching probability influence the algorithm's exploration and exploitation balance.

    Uses and Applications 🛠️

    The HBA can be applied to various optimization problems, including:

        Function Optimization: Finding the minimum or maximum of mathematical functions.
        Engineering Design: Optimizing parameters for engineering systems.
        Machine Learning: Hyperparameter tuning for models.
        Feature Selection: Identifying the most relevant features in a dataset.

    """
    )
    return


@app.cell
def _():
    import numpy as np

    def fitness(x):
      return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

    # === params ===
    num_badgers = 100
    num_dims = 2
    max_iter = 1000
    alpha = 0.01 # digging depth factor.
    beta = 0.1 # hunting radius factor.
    p = 0.5 # Switching probability between digging and hunting.
    lower_bound = -10
    upper_bound = 10
    # === initalize random values ===
    positions = np.random.uniform(lower_bound, upper_bound, (num_badgers, num_dims))
    fitness_values = np.array([fitness(pos) for pos in positions])
    best_position = positions[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    # === initalize food source positions and values ======
    food_sources = np.tile(best_position, (num_badgers, 1))
    food_fitness = np.tile(best_fitness, (num_badgers))

    # === main loop ====
    for i in range(max_iter):
      # loop over each honey badger
      for j in range(num_badgers):
        # Choose a random food source index
        food_index = np.random.randint(num_badgers)
        # Calculate the distance between the badger and the food source
        distance = np.linalg.norm(positions[j] - food_sources[food_index])
        # Calculate the fitness difference between the badger and the food source
        q = np.random.rand()
        if q < p:
          # Use digging model to update position.
          new_position = distance * np.exp(-alpha * distance) * np.cos(2 * np.pi * distance) + food_sources[food_index]
        else:
          # Use hunting model to update position.
          new_position = food_sources[food_index] + beta * distance * np.random.uniform(-1, 1, num_dims)
        # Apply boundary conditions.
        new_position = np.clip(new_position, lower_bound, upper_bound)
        # Calculate the fitness of the new position
        new_fitness = fitness(new_position)
        # Update the badger's position and fitness if the new fitness is better
        if new_fitness < fitness_values[j]:
          positions[j] = new_position.copy()
          fitness_values[j] = new_fitness
          if new_fitness < best_fitness:
            best_position = new_position.copy()
            best_fitness = new_fitness
        # Sort the honey badgers by their fitness values in ascending order.
        sorted_indices = np.argsort(fitness_values)
        positions = positions[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        # Update the number of food sources based on a linear decreasing scheme
        num_food_sources = num_badgers - i * (num_badgers - 1) // max_iter
        # Update the food source positions and fitness values by copying from the best honey badgers.
        food_sources[:num_food_sources] = positions[:num_food_sources].copy()
        food_fitness[:num_food_sources] = fitness_values[:num_food_sources].copy()
    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
    return


if __name__ == "__main__":
    app.run()
