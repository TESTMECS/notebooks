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

    Overivew ✳️

        Population of slime molds that can explore a discrete or continuous search space.
        A diffusion model that simulates the movement and interaction of the slime molds.
        A chemotaxis mdoel that simulates attraction and repulsion of the slime molds by chemical signals
        A set of parameters that control the diffusion rate, chemotaxis sensitivity and population size.

    Uses

        The SMA can be used to find optimal solution to a complex optimization problem, such as routing networks or image segmentation.
        The diffusion model allows the slime molds to spread and merge with each other while the chemotaxis model allows them to move towards or away from chemical signals that represent the objective function.
        The algorithm terminates when a predefined criterion is met.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    def fitness(x):
      return np.sum(x**2)

    # === params ====
    num_slimes = 100
    num_dims = 2
    max_iter = 200
    diffusion_rate = 0.1
    chemotaxis_rate = 0.1 # Chemotaxis sensitvity
    lower_bound = -5
    upper_bound = 5
    # ===================
    positions = np.random.uniform(lower_bound, upper_bound, (num_slimes, num_dims))
    concentrations = np.random.uniform(0, 1, (num_slimes))

    # === main loop ======
    for i in range(max_iter):
      for j in range(num_slimes):
        # Random neighbor
        neighbor = j
        while neighbor == j:
          neighbor = np.random.randint(0, num_slimes)
        # find the difference between the current slime mold and its neighbor.
        delta_fitness = fitness(positions[j]) - fitness(positions[neighbor])
        # Update the position by applying the diffusion and chemotaxis models
        new_position = positions[j] + diffusion_rate * (positions[neighbor] - positions[j]) + chemotaxis_rate * delta_fitness * np.random.uniform(-1, 1, num_dims)
        # Apply boundary conditions
        new_position = np.clip(new_position, lower_bound, upper_bound)
        # Update the concentration by applying the diffusion model
        new_concentration = concentrations[j] + diffusion_rate * (concentrations[neighbor] - concentrations[j])
        # Update the positions and concentrations
        positions[j] = new_position.copy()
        concentrations[j] = new_concentration
        # Find the best slime mold and its fitness value
        best_slime = np.argmin([fitness(p) for p in positions])
        best_fitness = fitness(positions[best_slime])
        #print(f"Iteration {i}, Best fitness: {best_fitness}")
    print(f"Best slime mold: {positions[best_slime]}")
    print(f"Best fitness: {best_fitness}")

    return


if __name__ == "__main__":
    app.run()
