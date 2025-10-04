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
    🔥 Moth-Flame Optimization (MFO) Algorithm
    Overview

    Moth-Flame Optimization (MFO) is a nature-inspired optimization algorithm that mimics the navigation behavior of moths in the presence of flames. The core concepts of MFO are:

        Population of Moths: These represent candidate solutions in the search space.
        Flames: These represent optimal or near-optimal solutions found so far.
        Spiral Equation: A mathematical model used to simulate the movement of moths around flames. This equation balances exploration and exploitation during the search process.
        Multiple Flames: The algorithm utilizes multiple flames to explore different regions of the search space. The number of flames is typically reduced over iterations to shift from exploration to exploitation.
        Flame Absorption Coefficients: These coefficients can be used to adjust the influence of flames on the moth's movement, further balancing exploration and exploitation.

    💡 Uses of MFO

    Moth-Flame Optimization is a versatile algorithm that can be applied to a wide range of optimization problems, including:

        Classification: Optimizing parameters of machine learning models.
        Clustering: Finding optimal cluster centers.
        Scheduling: Solving complex scheduling problems.
        Feature Selection: Identifying the most relevant features in a dataset.
        Engineering Design: Optimizing parameters in engineering applications.

    """
    )
    return


@app.cell
def _():
    import numpy as np
    import scipy

    def fitness(x):
      return np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0]- np.pi)**2 + (x[1] - np.pi)**2))

    # === params ====
    num_moths = 110
    num_dims = 2
    max_iter = 200
    gamma = 0.1 # Step size factor for spiral equation.
    lower_bound = -10
    upper_bound = 10

    # === initalize positions and fitness values randomly ========
    positions = np.random.uniform(lower_bound, upper_bound, (num_moths, num_dims))
    fitness_values = np.array([fitness(pos) for pos in positions])
    best_position = positions[np.argmax(fitness_values)]
    best_fitness = np.min(fitness_values)

    flames = np.tile(best_position, (num_moths, 1))
    flame_fitness = np.tile(best_fitness, num_moths)

    # === main loop ====
    for i in range(max_iter):
      # Loop over each moth.
      for j in range(num_moths):
        # Choose random flame index.
        flame_index = np.random.randint(num_moths)
        # Calculate the distance between the current moth and the chosen flame.
        distance = np.linalg.norm(positions[j] - flames[flame_index])
        # Generate a new position by applying the spiral equation around the chosen flame.
        new_position = distance * np.exp(-gamma * distance) * np.cos(2 * np.pi * distance) + flames[-flame_index]
        # Apply boundary conditions
        new_position = np.clip(new_position, lower_bound, upper_bound)
        # Evaluate the fitness value of the new position.
        new_fitness = fitness(new_position)
        # Compare the new fitness with the old one and update accordingly.
        if new_fitness > fitness_values[j]:
          positions[j] = new_position.copy()
          fitness_values[j] = new_fitness
          if new_fitness > best_fitness:
            best_position = new_position.copy()
            best_fitness = new_fitness
        # Store the moths by their fitness values in ascending order.
        sorted_indices = np.argsort(fitness_values)
        positions = positions[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        # Update the number of flames based on a linear decreasing scheme.
        num_flames = num_moths - i * (num_moths - 1 // max_iter)
        flames[:num_flames] = positions[:num_flames].copy()
        flame_fitness[:num_flames] = fitness_values[:num_flames].copy()
        #print(f"Iteration {i}, Best Fitness: {best_fitness}")
    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}")

    return


if __name__ == "__main__":
    app.run()
