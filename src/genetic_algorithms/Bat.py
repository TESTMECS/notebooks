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
    The Bat Algorithm: An Overview 🦇
    Introduction

    Inspired by the sophisticated echolocation behavior of microbats, the Bat Algorithm is a powerful metaheuristic developed for solving global optimization problems. It simulates a population of artificial bats navigating a multidimensional search space, mimicking the bats' ability to locate prey (optimal solutions) using sound pulses. The algorithm effectively balances exploration of the search space with exploitation of promising regions to efficiently find near-optimal or optimal solutions to complex problems.
    Core Concepts

    The Bat Algorithm is built upon several key concepts derived from the behavior of real bats:

        Echolocation: Bats emit ultrasonic pulses and listen for the echoes to determine the distance, size, and shape of objects. In the algorithm, this is represented by the generation of new candidate solutions.
        Frequency (f): Represents the speed at which a bat changes its position. Lower frequencies correspond to larger step sizes, promoting exploration, while higher frequencies lead to smaller steps and exploitation.
        Velocity (v): Determines the direction and magnitude of a bat's movement in the search space.
        Loudness (A): Represents the intensity of the emitted pulse. Initially high, loudness decreases as a bat finds better solutions, reducing the likelihood of generating new random solutions and favoring exploitation.
        Pulse Rate (r): Indicates the rate at which a bat emits pulses. Initially low, the pulse rate increases as a bat approaches a potential solution, increasing the likelihood of exploiting the best solution found so far.
        Global Best Solution: The best position found by any bat in the population up to the current iteration. This guides the search process.

    Applications and Uses

    The Bat Algorithm has proven to be effective in solving a wide range of optimization problems across various domains. Some common applications include:

        Function Optimization: Finding the global minimum or maximum of complex mathematical functions, including multimodal functions with numerous local optima.
        Engineering Design: Optimizing parameters in engineering problems, such as structural design, control systems, and electrical engineering.
        Machine Learning: Used for feature selection, hyperparameter tuning, and training neural networks.
        Image Processing: Employed in tasks like image segmentation and feature extraction.
        Scheduling and Logistics: Applied to solve problems related to scheduling tasks, vehicle routing, and resource allocation.

    The algorithm's ability to handle continuous search spaces and its balance between exploration and exploitation make it a versatile tool for optimization.
    """
    )
    return


@app.cell
def _(loudnesses):


    import numpy as np

    def fitness(x):
      return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    # === params =====
    num_bats = 100
    num_dims = 10
    max_iter = 200
    freq_min = 0.001
    freq_max = 0.1
    init_loudness = 1
    alpha = 0.9 # Loudness reduction factor
    r0 = 0.5 # Initial pulse rate
    gamma = 0.9 # Pulse Rate increase factor
    lower_bound = -500
    upper_bound = 500

    # === initalize bat positions, velocities, frequencies =====
    positions = np.random.uniform(lower_bound, upper_bound, (num_bats, num_dims))
    velocities = np.zeros((num_bats, num_dims))
    frequencies = np.random.uniform(freq_min, freq_max, (num_bats))
    loudness = np.ones(num_bats) * init_loudness
    pulse_rates = np.ones(num_bats) * r0
    fitness_values = np.array([fitness(x) for x in positions])
    # best position and its fitness values
    best_position = positions[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    # === main loop =============
    for i in range(max_iter):
      for j in range(num_bats):
        # Update frequency by applying the freq function
        frequencies[j] = freq_min + (freq_max - freq_min) * np.random.rand()
        # Update the velocity by adding the product of frequency and distance to best position
        velocities[j] = velocities[j] + (positions[j] - best_position) * frequencies[j]
        # Update the position by adding the velocity
        positions[j] = positions[j] + velocities[j]
        # Apply boundary conditions
        positions[j] = np.clip(positions[j], lower_bound, upper_bound)
        # Generate a new position by adding some randomness to the current position
        new_position = positions[j] + np.random.uniform(-1, 1, num_dims)
        # Apply the boundary conditions
        new_position = np.clip(new_position, lower_bound, upper_bound)
        # Evaluate the new fitness of the new position
        new_fitness = fitness(new_position)
        # Compare the new fitness with the old one and update accordingly.
        if new_fitness < fitness_values[j] and (np.random.rand() < loudness[j]):
          positions[j] = new_position.copy()
          fitness_values[j] = new_fitness

          if new_fitness < best_fitness:
            best_position = new_position.copy()
            best_fitness = new_fitness
        # Generate another new position by exploiting the best position if a random number is less than pulse rate
        if np.random.rand() < pulse_rates[j]:
          # Generate a new position by adding some randomness to best position
          new_position = best_position + np.random.uniform(-1, 1, num_dims) * loudnesses[j]
          # Apply the boundary conditions
          new_position = np.clip(new_position, lower_bound, upper_bound)
          # Evaluate the new fitness of the new position
          new_fitness = fitness(new_position)
          # compare the new fitness with the old one and update accordingly if a random number is less than loudness value
          if (new_fitness < fitness_values[j]) and (np.random.rand() < loudnesses[j]):
            positions[j] = new_position.copy()
            fitness_values[j] = new_fitness
            # Update the loudness and pulse rate values
            loudness[j] = alpha * loudness[j]
            pulse_rates[j] = r0 * (1 - np.exp(-gamma * i))
          else:
            # Update the loudness and pulse rate values
            loudness[j] = alpha * loudness[j]
            pulse_rates[j] = r0 * (1 - np.exp(-gamma * i))

            # Update the best position and fitness if improved
            if new_fitness < best_fitness:
              best_position = positions[j].copy()
              best_fitnesss = new_fitness

            #print(f"Iteration {i+1}: Best fitness = {best_fitness:.4f}")
    print(f"Best solution: {best_position}")
    print(f"Best fitness: {best_fitness}")
    return


if __name__ == "__main__":
    app.run()
