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
    🐳 Whale Optimization Algorithm (WOA)
    Overview ✨

    The Whale Optimization Algorithm (WOA) is a metaheuristic optimization algorithm inspired by the hunting behavior of humpback whales. It is based on the bubble-net feeding strategy employed by these fascinating creatures. The algorithm simulates two main behaviors:

        Encircling Prey: Whales identify and surround their prey. The algorithm mimics this by having search agents (representing whales) update their positions towards the current best solution found so far.
        Spiral Bubble-net Attacking: Whales create a spiral-shaped bubble net to trap prey. The algorithm models this by a spiral movement of search agents towards the best solution.

    A key aspect of the algorithm is the trade-off between exploration (searching broadly) and exploitation (refining solutions around good candidates), controlled by adaptive parameters.
    Uses 🛠️

    The Whale Optimization Algorithm is a versatile tool that can be applied to a wide range of complex optimization problems across various domains. Some common uses include:

        Global Optimization: Finding the optimal solution in continuous search spaces for functions with multiple local optima.
        Feature Selection: Identifying the most relevant features in machine learning tasks to improve model performance and reduce dimensionality.
        Parameter Tuning: Optimizing hyperparameters of machine learning models or other algorithms.
        Engineering Design: Solving complex design problems in areas like structural engineering, electrical engineering, and mechanical engineering.
        Data Clustering: Finding optimal cluster centers and assignments in data analysis.

    """
    )
    return


@app.cell
def _():
    import numpy as np

    # Levy function.
    def fitness(x):
      return np.sin(3*np.pi*x[0])**2 + np.sum((x[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * x[:-1] + 1)**2)) + (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)

    # === Params =====
    num_whales = 100
    num_dims = 10
    max_iter = 1000
    a_max = 2 # Maximum value of a parameter
    a_min = 0 # Minimum value of a parameter
    b = 1 # Constant for spiral model.
    lower_bound = -10
    upper_bound = 10

    # === Initalize the fitness values randomly =======
    positions = np.random.uniform(lower_bound, upper_bound, size=(num_whales, num_dims))
    fitness_values = np.array([fitness(x) for x in positions])
    best_position = positions[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    # === main loop ==========
    for i in range(max_iter):
      for j in range(num_whales):
        # Generate a random parameter between 0 and 1
        r = np.random.rand()
        # Update the value of a parameter linearly from a_max to a_min
        a = a_max - i * (a_max - a_min) / max_iter
        # Calculate the value of c parameter between -1 and 1
        c = 2 * r
        # Calculate the distance between the current whale and the best whale
        d = np.abs(c*best_position - positions[j])
        # Choose either spiral model or encircling model based on a random parameter p between 0 and 1
        p = np.random.rand()

        if p < 0.5:
          # Spiral model
          l = np.random.uniform(-1, 1)
          new_position = d * np.exp(l * b) * np.cos(2 * np.pi * l) + best_position
        else:
          # encircling model to update position
          new_position = best_position - a * d
          # Apply boundary conditions
          new_position = np.clip(new_position, lower_bound, upper_bound)
          # Evaluate the fitness value of the new position
          new_fitness = fitness(new_position)
          # Update the best position and fitness value if the new position is better
          if new_fitness < fitness_values[j]:
            positions[j] = new_position.copy()
            fitness_values[j] = new_fitness
            # Update the best position and fitness if improved
            if new_fitness < best_fitness:
              best_position = positions[j].copy()
              best_fitness = new_fitness
          print(f"Iteration {i+1}/{max_iter}, Best fitness: {best_fitness}, Best position: {best_position}")
    print(f"Best fitness: {best_fitness}, Best position: {best_position}")
    print(f"Best fitness: {best_fitness}, Best position: {best_position}")
    return


if __name__ == "__main__":
    app.run()
