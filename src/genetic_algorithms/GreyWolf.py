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
    🐺 Grey Wolf Optimization (GWO) Algorithm 🐺
    Overview

        Inspired by the social hierarchy and hunting behavior of grey wolves.
        Population of grey wolves that hunt in the continuous search space.
        A Leader wolf (alpha) that guides the pack towards promising areas.
        Follower wolves (beta and delta) that help scout the search space.
        Omega wolves that follow the guidance of the alpha, beta, and delta wolves.
        A set of encircling, hunting, and attacking coefficents to balance exploration vs explotation.

    Uses 🎯

        Find the optimal solution to complex optimization problems like function optimization, feature selection, scheduling etc.
        At each iteration the alpha, beta, and delta wolves guide the omega wolves by updating their distance and position based on the best three solutions.
        The omega wolves update their positions by encircling, hunting, and attacking the prey guided by the top three wolves.

    """
    )
    return


@app.cell
def _():
    import numpy as np

    def fitness(x):
      return -np.cos(x[0]) * np.cos(x[1]) * np.exp(- ((x[0] - np.pi)**2 + (x[1] - np.pi)**2 ))

    num_wolves = 50
    num_dims = 2
    max_iter = 200
    # Parameters for encircling prey.
    A1 = 2
    A2 = 2
    A3 = 2
    A4 = 2
    lower_bound = -10
    upper_bound = 10

    # === initalize positions and values randomly =====
    positions = np.random.uniform(lower_bound, upper_bound, (num_wolves, num_dims))
    fitness_values = np.array([fitness(pos) for pos in positions])
    best_positions = positions[np.argmin(fitness_values)]
    best_fitness_values = np.min(fitness_values)

    # === main loop ====
    for i in range(max_iter):
      # loop over each wolf.
      for j in range(num_wolves):
        # Choose a random alpha, beta, and delta wolf from the population.
        alpha, beta, delta = np.random.choice(num_wolves, 3, replace=False)
        # Calculate the distance between the current wolf and the alpha, beta, and delta.
        dist_alpha = np.linalg.norm(positions[j] - positions[alpha])
        dist_beta = np.linalg.norm(positions[j] - positions[beta])
        dist_delta = np.linalg.norm(positions[j] - positions[delta])
        # Generate a new position by encircling the alpha, beta, and delta wolves.
        x1 = positions[alpha] - A1 * dist_alpha * np.random.rand()
        x2 = positions[beta] - A2 * dist_beta * np.random.rand()
        x3 = positions[delta] - A3 * dist_delta * np.random.rand()
        # New position is an average.
        x_hunt = (x1 + x2 + x3) / 3
        # Generate a new position by attacking alpha wolf with more randomness.
        X_attack = x_hunt - A4 * (np.random.rand() - 0.5) * dist_alpha
        # Apply boundary conditions
        X_attack = np.clip(X_attack, lower_bound, upper_bound)
        # Evaluate the fitness of the new position.
        new_fitness = fitness(X_attack)
        # Compare the new fitness with the old one and update accordingly
        if new_fitness < fitness_values[j]:
          positions[j] = X_attack.copy()
          fitness_values[j] = new_fitness
          if new_fitness < best_fitness_values:
            best_fitness_values = new_fitness
            best_positions = positions[j].copy()

    print("Best position: ", best_positions)
    print("Best fitness: ", best_fitness_values)
    return


if __name__ == "__main__":
    app.run()
