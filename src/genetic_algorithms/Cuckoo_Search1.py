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
    Cuckoo Search Algorithm 🐦
    Concepts ✨

    The Cuckoo Search (CS) algorithm is a nature-inspired metaheuristic optimization algorithm. It is based on the obligate brood parasitism of some cuckoo species that lay their eggs in the nests of other host birds. The core concepts include:

        Cuckoos and Eggs: Each cuckoo represents a potential solution to an optimization problem, and its egg represents the candidate solution.
        Host Nests: These represent possible locations for the cuckoo eggs (candidate solutions).
        Levy Flights: Cuckoos search for host nests using Levy flights, which are random walks with occasional long jumps. This helps explore the search space effectively.
        Egg Discovery: Host birds may discover the foreign cuckoo eggs with a certain probability and either throw them away or abandon the nest and build a new one. This mechanism helps in discarding poor solutions and generating new ones.

    Uses 💡

    The Cuckoo Search algorithm is a versatile optimization method that can be applied to various problems, including:

        Solving complex optimization problems: CS has been successfully used to find optimal solutions for challenging problems, including multimodal functions and engineering design optimization.
        Feature selection: It can be used to select the most relevant features for machine learning models.
        Scheduling and resource allocation: CS can help in optimizing schedules and allocating resources efficiently.
        Image processing: It has applications in image segmentation and other image processing tasks.

    """
    )
    return


@app.cell
def _():
    import numpy as np
    import scipy

    # Michalewicz
    def fitness(x):
      return -np.sum(np.sin(x) * np.sin((np.arange(1, len(x)+1) * x**2) / np.pi)**20)

    # === params =====
    num_cuckoos = 100
    num_dims = 10
    max_iter = 500
    alpha = 0.01 # Levy flight scale factor.
    beta = 1.5 # Levy flight exponent.
    p = 0.25 # Discovery rate.
    lower_bound = 0
    upper_bound = np.pi

    # === initalize the cuckoo positions and fitness values randomly ====
    positions = np.random.uniform(lower_bound, upper_bound, (num_cuckoos, num_dims))
    fitness_values = np.array([fitness(x) for x in positions])
    best_position = positions[np.argmax(fitness_values)]
    best_fitness = np.min(fitness_values)

    # === main loop =======
    for i in range(max_iter):
      # Loop over each cuckoo
      for j in range(num_cuckoos):
        # Generate a new position by applying levy flight model
        new_position = positions[j] + alpha * scipy.stats.levy_stable.rvs(beta, 0, size=num_dims)
        # Apply boundary conditions
        new_position = np.clip(new_position, lower_bound, upper_bound)
        # Evaluate the fitness of the new position
        new_fitness = fitness(new_position)
        # Compare the new fitness with the old one and update accordingly
        if new_fitness > fitness_values[j]:
          positions[j] = new_position
          fitness_values[j] = new_fitness
          if new_fitness > best_fitness:
            best_position = new_position.copy()
            best_fitness = new_fitness
          # Abandon a fraction of the worst cuckoos and replace them with new ones randomly generated.
          num_abandoned = int(p * num_cuckoos)
          worst_indices = np.argsort(fitness_values)[-num_abandoned:]
          positions[worst_indices] = np.random.uniform(lower_bound, upper_bound, (num_abandoned, num_dims))
          fitness_values[worst_indices] = np.array([fitness(x) for x in positions[worst_indices]])
          #print(f"Iteration {i+1}: Best fitness = {best_fitness}")
    print(f"Best solution: {best_position}")
    print(f"Best fitness: {best_fitness}")
    return


if __name__ == "__main__":
    app.run()
