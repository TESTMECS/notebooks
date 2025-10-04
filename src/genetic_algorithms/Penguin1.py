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

    🐧 Penguin Search Optimization (PSO) Algorithm
    Overview ✨

        Population of penguins that can dive and swim in a continuous search space.
        A prey matrix that stores the position and fitness value of each prey (potential solutions).
        A diving model that simulates the penguins' movement towards the prey.
        A swimming model that simulates the penguins' movement around the prey.
        A set of parameters that control the diving depth, the swimming radius, and the number of prey.

    Uses 🚀

        At each iteration, each penguin updates its position by applying either the diving model or the swimming model, depending on a random parameter.
        Penguins move towards the swimming model in a spiral shape.
        Also allows the penguin to move around the best solution randomly.
        The prey are updated by better penguin solutions found during the iteration.

    Future Research Directions 🔬

        Adapting the algorithm for discrete optimization problems.
        Exploring different variations of the diving and swimming models.
        Investigating the impact of different parameter settings on performance.
        Applying the algorithm to real-world optimization problems in various domains.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    def fitness(x):
      return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] -np.pi)**2))

    # === params ====
    num_penguins = 50
    num_dims = 2
    max_iter = 200
    alpha = 0.01 # Diving depth factor
    beta = 0.1 # Swimming Radius factor
    p = 0.5 # Switching probability between diving and swimming.
    lower_bound = -10
    upper_bound = 10

    # === initalize the penguin positions and fitness values randomly ====
    positions = np.random.uniform(lower_bound, upper_bound, size=(num_penguins, num_dims))
    fitness_values = np.array([fitness(pos) for pos in positions])
    best_position = positions[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)

    # === initalize prey positions. =====
    prey = np.tile(best_position, (num_penguins, 1))
    prey_fitness = np.tile(best_fitness, num_penguins)

    # === main loop ===
    for i in range(max_iter):
      # loop over each penguin
      for j in range(num_penguins):
        # choose a random prey index.
        prey_index = np.random.randint(num_penguins)
        # calculate the distance between the penguin and the prey.
        distance = np.linalg.norm(positions[j] - prey[prey_index])
        # Choose either diving model or swimming model based a random parameter q between 0 and 1.
        q = np.random.rand()
        if q < p:
          # Use diving model to update position.
          new_position = distance * np.exp(-alpha * distance) * np.cos(2 * np.pi * distance) + prey[prey_index]
        else:
          # Use swimming model to update position.
          new_position = prey[prey_index] + beta * distance * np.random.uniform(-1, 1, num_dims)
        # Apply boundary conditions
        new_position = np.clip(new_position, lower_bound, upper_bound)
        # Evaluate new fitness.
        new_fitness = fitness(new_position)
        # Update prey if new fitness is better.
        if new_fitness < fitness_values[j]:
          positions[j] = new_position.copy()
          fitness_values[j] = new_fitness
          if new_fitness < best_fitness:
            best_position = new_position.copy()
            best_fitness = new_fitness
        # Sort the penguins by their fitness values in ascending order.
        sorted_indices = np.argsort(fitness_values)
        positions = positions[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        # Update the number of prey based on a linear decreasing scheme.
        num_prey = num_penguins - i * (num_penguins - 1) // max_iter
        # Update the prey positions and fitness values by copying from the best penguins.
        prey[:num_prey] = positions[:num_prey].copy()
        prey_fitness[:num_prey] = fitness_values[:num_prey].copy()

    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
    return


if __name__ == "__main__":
    app.run()
