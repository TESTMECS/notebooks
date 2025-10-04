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
    🌸 Flower Pollination Algorithm (FPA)

    The Flower Pollination Algorithm (FPA) is a nature-inspired metaheuristic optimization algorithm developed by Xin-She Yang in 2012. It mimics the fascinating process of flower pollination, where flowers reproduce by transferring pollen.
    💡 Concepts:

        Pollination Process: FPA is inspired by the two main types of pollination:
            Global Pollination: Simulates the long-distance movement of pollen by wind or insects, often involving a Lévy flight distribution for larger jumps in the search space.
            Local Pollination: Models the short-distance movement of pollen, like self-pollination or pollination by nearby insects, typically using a local random walk.
        Flower Constancy: This concept suggests that pollinators tend to visit flowers of the same species, which can be related to the selection pressure towards better solutions.
        Switching Probability (p): A crucial parameter that controls the likelihood of switching between global and local pollination. A higher 'p' favors global exploration, while a lower 'p' emphasizes local exploitation.

    🛠️ Uses:

    FPA has been successfully applied to a wide range of optimization problems due to its balance between exploration and exploitation:

        Engineering Design: Optimizing parameters in various engineering fields, such as structural design, electrical circuits, and control systems.
        Machine Learning: Parameter tuning for machine learning models and feature selection.
        Image Processing: Image segmentation and edge detection.
        Job Scheduling: Optimizing task allocation and scheduling in complex systems.
        Solving Complex Optimization Problems: Particularly effective for multimodal functions and problems with many local optima.

    """
    )
    return


@app.cell
def _():
    import numpy as np
    import scipy

    def fitness(x):
      return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

    # === params ======
    num_flowers = 100
    num_dims = 2
    max_iter = 200
    p = 0.8 # switching probability between global and local pollination.
    gamma = 0.1 # Step size factor for global pollination.
    lower_bound = -100
    upper_bound = 100
    # ==================
    positions = np.random.uniform(lower_bound, upper_bound, (num_flowers, num_dims))
    fitness_values = np.array([fitness(pos) for pos in positions])
    best_position = positions[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    # === main loop ====
    for i in range(max_iter):
      # Loop over each flower
      for j in range(num_flowers):
        # Generate a random number
        q = np.random.rand()
        if q < p:
          # Global pollination to update position.
          new_position = positions[j] + gamma * scipy.stats.levy_stable.rvs(1.5, 0, size=num_dims) * (best_position - positions[j])
        else:
          # Local pollination to update position.
          neighbor = j
          while neighbor == j:
            neighbor = np.random.randint(num_flowers)
          new_position = positions[j] + np.random.uniform(-1, 1, num_dims) * (positions[j] - positions[neighbor])
          # Apply boundary conditions
          new_position = np.clip(new_position, lower_bound, upper_bound)
          # Evaluate the fitness value of the new position.
          new_fitness = fitness(new_position)
          # Update the position and fitness value if the new fitness value is better.
          if new_fitness < fitness_values[j]:
            positions[j] = new_position.copy()
            fitness_values[j] = new_fitness
            if new_fitness < best_fitness:
              best_position = positions[j].copy()
              best_fitness = new_fitness
          #print(f"Iteration {i+1}: Best fitness = {best_fitness:.4f}")
    print(f"Best position: {best_position}")
    print(f"Best fitness: {best_fitness:.4f}")
    return


if __name__ == "__main__":
    app.run()
