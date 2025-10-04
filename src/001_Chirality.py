# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "rich==14.0.0",
#     "torch==2.7.1",
# ]
# ///

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
    **The ChiralNet: From Biological Analogy to a Universal "Differential Engine"**

    **Goal:** To demonstrate how a simple, bio-inspired neural architecture evolved into a powerful scientific instrument for measuring the structural relationships within data, capable of tackling famously hard problems in science and computer science.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""> Chirality is a property of an object or system that cannot be superimposed on its mirror image — like your left and right hands. The term comes from the Greek word cheir, meaning hand."""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Why it Matters 🥇
    Chirality shows up in life, the universe, and computation. It often signals a fundamental asymmetry. Something about the system "chooses a side" and that has profound implications for how we understand and interact with the world.
    """
    )
    return


@app.cell
def _():
    import logging

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rich.logging import RichHandler

    return F, RichHandler, logging, nn, plt, torch


@app.cell
def _(RichHandler, logging, torch):
    LOGLEVEL = "info".upper()
    logging.basicConfig(level=LOGLEVEL, format="%(message)s", handlers=[RichHandler()])
    cout = logging.getLogger("rich")
    cout.info("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cout.info(f"Using device: {device}")
    return cout, device


@app.cell
def _(F, mo, nn):
    mo.md("#--- Model Definition ---")

    class ChiralNeuronLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.left = nn.Linear(dim, dim)
            self.right = nn.Linear(dim, dim)

        def forward(self, x):
            l_out = F.relu(self.left(x))
            r_out = F.relu(self.right(-x))
            return l_out, r_out

    class ChiralNet(nn.Module):
        def __init__(self, dim=4):
            super().__init__()
            # Using a single, wider chiral layer for the 4D task
            hidden_dim = 64
            self.chiral = ChiralNeuronLayer(dim)
            self.fc_left = nn.Linear(dim, hidden_dim)
            self.fc_right = nn.Linear(dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            l, r = self.chiral(x)
            # Process the difference through separate MLPs to give the model more capacity
            l_out = F.relu(self.fc_left(l))
            r_out = F.relu(self.fc_right(r))

            # The competitive interaction remains
            out = self.output(l_out - r_out)
            return out, l.norm(), r.norm()

    return (ChiralNet,)


@app.cell
def _(cout, torch):
    # --- Data Generation ---
    def get_8d_data(n=512):
        x = torch.randn(n, 8) * 2
        center = torch.randn(1, 8) * 0.5
        radius_inner = 1.5
        radius_outer = 2.5
        distance_sq = torch.sum((x - center) ** 2, dim=1)
        y = (
            ((distance_sq > radius_inner**2) & (distance_sq < radius_outer**2))
            .float()
            .unsqueeze(1)
        )
        return x, y

    cout.info(f"x: {get_8d_data()[0].shape} \n y: {get_8d_data()[1].shape}")

    return (get_8d_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Simple Chiral Neuron.
    ```python
    class ChiralNeuronLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.left = nn.Linear(dim, dim)
            self.right = nn.Linear(dim, dim)

        def forward(self, x):
            l_out = F.relu(self.left(x))
            r_out = F.relu(self.right(-x))
            return l_out, r_out

    class ChiralNeuronLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.left = nn.Linear(dim, dim)
            self.right = nn.Linear(dim, dim)

        def forward(self, x):
            l_out = F.relu(self.left(x))
            r_out = F.relu(self.right(-x))
            return l_out, r_out
    ```
    ## Explaination
    - In biology only one representation 'wins' therefore what we will do is create a symmetric neuron that encodes both $x$ and its inverse $-x$. Then we will simply subtract $x$ from $-x$ to get the output which should tell us how much the symmetric neuron balances the original $x$ or the inverse representation as it tries to solve tasks $-x$. Could this be a general measure of "symmetry"?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""Training the `ChiralityNet` model to predict points in the torus inside in 8D space."""
    )
    return


@app.cell
def _(ChiralNet, device, get_8d_data, mo, nn, torch):
    def _():
        # --- Configuration ---
        torch.manual_seed(42)
        epochs = 100
        lr = 0.01

        # 1. Setup model
        model = ChiralNet(dim=8).to(device)
        # 2. Add weight decay for better regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        # 3. Generate data and move it to the GPU
        x_train, y_train = get_8d_data(
            n=2048
        )  # Increased data size for better generalization
        x_train, y_train = x_train.to(device), y_train.to(device)
        # --- Training Loop ---
        left_norms = []
        right_norms = []
        losses = []
        model.train()  # Set model to training mode
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, l_norm, r_norm = model(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
            left_norms.append(l_norm.item())
            right_norms.append(r_norm.item())
            losses.append(loss.item())

        mo.output.append("\n--- Training on 8D Data ---")
        mo.output.append(f"Final Loss: {losses[-1]}")
        mo.output.append(f"Final Left Norm: {left_norms[-1]} ")
        mo.output.append(f"Final Right Norm: {right_norms[-1]}")
        # ----- Testing -----------
        x_test, y_test = get_8d_data(n=1024)
        x_test, y_test = x_test.to(device), y_test.to(device)

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            test_output, _, _ = model(x_test)
            test_loss = loss_fn(test_output, y_test)
            test_preds = torch.sigmoid(test_output) > 0.5
            test_accuracy = (test_preds == y_test).float().mean()

        mo.output.append("\n--- Evaluation on Separate 8D Test Set ---")
        mo.output.append(f"Test Loss: {test_loss.item()}")
        mo.output.append(
            f"Test Accuracy: {test_accuracy.item()} = {100 * test_accuracy.item()}%"
        )
        return x_test, y_test, test_preds, left_norms, right_norms, losses

    x_test, y_test, test_preds, left_norms, right_norms, losses = _()
    return left_norms, losses, right_norms, test_preds, x_test, y_test


@app.cell
def _(left_norms, losses, mo, plt, right_norms, test_preds, x_test, y_test):
    def viz(xt, yt, tp, losses):
        # --- Visualization ---
        # Detach from GPU
        x_test_np = xt.cpu().detach().numpy()
        y_test_np = yt.cpu().detach().squeeze().numpy()
        test_preds_np = tp.cpu().detach().squeeze().numpy()
        # Visualize projection onto the first 3 dimensions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        correct_inside = (test_preds_np == 1) & (y_test_np == 1)
        ax.scatter(
            x_test_np[correct_inside, 0],
            x_test_np[correct_inside, 1],
            x_test_np[correct_inside, 2],
            color="blue",
            label="Correct Inside (Proj)",
            alpha=0.6,
            marker="o",
        )
        correct_outside = (test_preds_np == 0) & (y_test_np == 0)
        # NOTE: Change color here to 'gray' see torus points.
        ax.scatter(
            x_test_np[correct_outside, 0],
            x_test_np[correct_outside, 1],
            x_test_np[correct_outside, 2],
            color="red",
            label="Correct Outside (Proj)",
            alpha=0.4,
            marker="o",
        )
        incorrect = test_preds_np != y_test_np
        ax.scatter(
            x_test_np[incorrect, 0],
            x_test_np[incorrect, 1],
            x_test_np[incorrect, 2],
            color="purple",
            label="Misclassified (Proj)",
            alpha=1.0,
            marker="X",
            s=100,
        )
        ax.set_title("Model Predictions on Test Set (Projection onto first 3D)")
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        ax.set_zlabel("Dimension 2")
        ax.legend()
        plt.tight_layout()

        # Plotting chirality dominance for the 8D dataset
        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(left_norms, label="Left Path Norm (8D)")
        plt.plot(right_norms, label="Right Path Norm (8D)")
        plt.title("Chiral Dominance Over Epochs (8D Dataset)")
        plt.xlabel("Epoch")
        plt.ylabel("Pathway Norm")
        plt.legend()
        plt.grid(True)

        # Plotting loss for the 8D dataset
        fig3 = plt.figure()
        plt.plot(losses)
        plt.title("Loss Over Epochs (8D Dataset)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        mo.output.append(fig)
        mo.output.append(fig2)
        mo.output.append(fig3)
        return

    viz(x_test, y_test, test_preds, losses)
    return


@app.cell
def _(losses, mo):
    mo.md(
        f"""
    # What you are seeing. 

    1. The first graph shows both the correctly classified points inside the torus (blue) and outside the torus (red), along with misclassified points (purple) (0). 
    2. The Second graph shows why this is the case. So the Left Path Norm is consistently smaller than the Right path Norm. In this way, the model is learning that the "Torus" transformation from the loss is the "correct" one, and the model is learning to "choose a side" in the chiral sense. But we don't see a large symmetric break essentially because y is dependent on the random points that `x = torch.randn(n, 8) * 2` produces! 
    3. The third graph shows the loss drop to {losses[-1]} over training.

    Now for a twist, how would the model preform on a dataset that isn't as clearly seperated, like a twist.
    """
    )
    return


@app.cell
def _(cout, torch):
    def get_4d_chiral_data(n=2048):
        x = torch.randn(n, 4) * 2
        # Define a "chiral" property: a simplified curl or "twist".
        # The label is 1 if it "twists" one way.
        twist = x[:, 0] * x[:, 3] - x[:, 1] * x[:, 2]
        y = (twist > 0.5).float().unsqueeze(1)
        return x, y

    cout.info(
        f"x: {get_4d_chiral_data()[0].shape} \n y: {get_4d_chiral_data()[1].shape}"
    )
    return (get_4d_chiral_data,)


@app.cell
def _(ChiralNet, cout, device, get_4d_chiral_data, mo, nn, plt, torch):
    def _():
        epochs = 1000
        lr = 0.01
        model = ChiralNet(dim=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()
        x_train, y_train = get_4d_chiral_data()
        x_train, y_train = x_train.to(device), y_train.to(device)
        # --- Training Loop ---
        left_norms = []
        right_norms = []
        losses = []

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, l_norm, r_norm = model(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                left_norms.append(l_norm.item())
                right_norms.append(r_norm.item())
                losses.append(loss.item())
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        cout.info("\n--- Training on 4D Chiral Data ---")
        cout.info(f"Final Loss: {losses[-1]:.6f}")
        cout.info(f"Final Left Norm: {left_norms[-1]:.2f}")
        cout.info(f"Final Right Norm: {right_norms[-1]:.2f}")

        # --- Evaluation ---
        x_test, y_test = get_4d_chiral_data(n=1024)
        x_test, y_test = x_test.to(device), y_test.to(device)

        model.eval()
        with torch.no_grad():
            test_output, _, _ = model(x_test)
            test_loss = loss_fn(test_output, y_test)
            test_preds = torch.sigmoid(test_output) > 0.5
            test_accuracy = (test_preds == y_test).float().mean()

        cout.info("\n--- Evaluation on Separate 4D Test Set ---")
        cout.info(f"Test Loss: {test_loss.item():.4f}")
        cout.info(f"Test Accuracy: {test_accuracy.item():.4f}")
        # --- Visualization ---
        # 1. Plot Chiral Dominance
        fig = plt.figure(figsize=(12, 5))
        plt.plot(left_norms, label="Left Path Norm (4D)")
        plt.plot(right_norms, label="Right Path Norm (4D)")
        plt.title("Chiral Dominance Over Epochs (4D 'Twist' Dataset)")
        plt.xlabel("Epochs (x100)")
        plt.ylabel("Pathway Norm")
        plt.legend()
        plt.grid(True)

        # 2. Plot Loss
        fig2 = plt.figure(figsize=(12, 5))
        plt.plot(losses)
        plt.title("Loss Over Epochs (4D 'Twist' Dataset)")
        plt.xlabel("Epochs (x100)")
        plt.ylabel("Loss")
        plt.yscale("log")  # Log scale is useful for seeing loss drop to very low values
        plt.grid(True)

        # 3. Visualize the learned decision boundary on a 2D slice
        print("\nVisualizing learned decision boundary...")
        model.eval()

        # Create a grid in the x0, x2 plane, holding x1 and x3 constant
        n_grid = 100
        x_range = torch.linspace(-4, 4, n_grid)
        grid_x0, grid_x2 = torch.meshgrid(x_range, x_range, indexing="ij")

        # Hold other dimensions constant, e.g., x1=1 and x3=1
        grid_x1 = torch.ones_like(grid_x0)
        grid_x3 = torch.ones_like(grid_x0)

        # The ground truth boundary for this slice is: x0*1 - 1*x2 = 0.5  =>  x2 = x0 - 0.5
        # This is a straight diagonal line.

        grid_points = torch.stack(
            [
                grid_x0.flatten(),
                grid_x1.flatten(),
                grid_x2.flatten(),
                grid_x3.flatten(),
            ],
            dim=1,
        ).to(device)

        with torch.no_grad():
            grid_output, _, _ = model(grid_points)
            grid_preds = torch.sigmoid(grid_output).cpu().reshape(n_grid, n_grid)

        fig3 = plt.figure(figsize=(8, 7))
        plt.contourf(grid_x0, grid_x2, grid_preds, levels=20, cmap="RdBu_r", alpha=0.8)
        plt.colorbar(label="Predicted Probability (Twist > 0.5)")

        # Plot the true decision boundary for this slice
        true_boundary_x0 = x_range.numpy()
        true_boundary_x2 = true_boundary_x0 - 0.5
        plt.plot(
            true_boundary_x0,
            true_boundary_x2,
            "k--",
            linewidth=3,
            label="True Boundary (x₂ = x₀ - 0.5)",
        )

        plt.title("Decision Boundary Slice (x₁=1, x₃=1)")
        plt.xlabel("Dimension 0")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        mo.output.append(fig)
        mo.output.append(fig2)
        mo.output.append(fig3)
        return

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # What you are seeing.
    1. The first graph shows the left and right path norms over training. We can see that the curves are a lot closer. This is because the model requires information from both to "grow together" as a function of the twist. 
    The decision boundry visiualization shows that the model almost perfectly learns the decision boundary between the two variables. The model learns to maintain about an equal amount of information from both left and right norms. 

    What if this could be a "measure of Symmetry" in the data? Let's try to break it with Asymmetrical data.
    """
    )
    return


@app.cell
def _(cout, torch):
    def get_mostly_symmetric_data(n=2048):
        x = torch.randn(n, 4) * 2
        symmetric_part = x[:, 0] * x[:, 3] - x[:, 1] * x[:, 2]  # Old 'twist'
        asymmetric_part = x.sum(dim=1)  # Old sum
        # Combine them, with the asymmetric part being small
        y = (symmetric_part + 0.1 * asymmetric_part > 0.5).float().unsqueeze(1)
        return x, y

    cout.info(
        f"x: {get_mostly_symmetric_data()[0].shape} \n y: {get_mostly_symmetric_data()[1].shape}"
    )
    return (get_mostly_symmetric_data,)


@app.cell
def _(ChiralNet, device, get_mostly_symmetric_data, mo, nn, plt, torch):
    def _():
        # --- Training Setup ---
        epochs = 1000
        lr = 0.01
        model = ChiralNet(dim=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        x_train, y_train = get_mostly_symmetric_data()
        x_train, y_train = x_train.to(device), y_train.to(device)

        # --- Training Loop ---
        left_norms = []
        right_norms = []
        losses = []
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, l_norm, r_norm = model(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                left_norms.append(l_norm.item())
                right_norms.append(r_norm.item())
                losses.append(loss.item())

        mo.output.append("--- Training on 4D ASYMMETRIC Data ---")
        mo.output.append(f"Final Loss: {losses[-1]:.6f}")
        mo.output.append(f"Final Left Norm: {left_norms[-1]:.4f}")
        mo.output.append(f"Final Right Norm: {right_norms[-1]:.4f}")

        # --- Evaluation ---
        x_test, y_test = get_mostly_symmetric_data(n=1024)
        x_test, y_test = x_test.to(device), y_test.to(device)

        model.eval()
        with torch.no_grad():
            test_output, _, _ = model(x_test)
            test_loss = loss_fn(test_output, y_test)
            test_preds = torch.sigmoid(test_output) > 0.5
            test_accuracy = (test_preds == y_test).float().mean()

        mo.output.append("\n--- Evaluation on Separate 4D Test Set ---")
        mo.output.append(f"Test Loss: {test_loss.item():.4f}")
        mo.output.append(f"Test Accuracy: {test_accuracy.item():.4f}")

        # --- Visualization ---

        # 1. Plot Chiral Dominance
        fig = plt.figure(figsize=(12, 5))
        plt.plot(left_norms, label="Left Path Norm")
        plt.plot(right_norms, label="Right Path Norm")
        plt.title("Chiral Pathway Norms on ASYMMETRIC Task")
        plt.xlabel("Epochs (x50)")
        plt.ylabel("Pathway Norm")
        plt.legend()
        plt.grid(True)

        # 2. Plot Loss
        fig2 = plt.figure(figsize=(12, 5))
        plt.plot(losses)
        plt.title("Loss Over Epochs (Asymmetric Task)")
        plt.xlabel("Epochs (x50)")
        plt.ylabel("Loss")
        plt.grid(True)

        # 3. Visualize the learned decision boundary on a 2D slice
        model.eval()

        # Create a grid in the x0, x1 plane, holding x2 and x3 constant (at 0)
        n_grid = 100
        x_range = torch.linspace(-4, 4, n_grid)
        grid_x0, grid_x1 = torch.meshgrid(x_range, x_range, indexing="ij")

        # Hold other dimensions constant at 0
        grid_x2 = torch.zeros_like(grid_x0)
        grid_x3 = torch.zeros_like(grid_x0)

        # The ground truth boundary for this slice is: x0 + x1 + 0 + 0 = 0.5  =>  x1 = 0.5 - x0
        # This is a straight diagonal line.

        grid_points = torch.stack(
            [
                grid_x0.flatten(),
                grid_x1.flatten(),
                grid_x2.flatten(),
                grid_x3.flatten(),
            ],
            dim=1,
        ).to(device)

        with torch.no_grad():
            grid_output, _, _ = model(grid_points)
            grid_preds = torch.sigmoid(grid_output).cpu().reshape(n_grid, n_grid)

        fig3 = plt.figure(figsize=(8, 7))
        plt.contourf(
            grid_x0.numpy(),
            grid_x1.numpy(),
            grid_preds.numpy(),
            levels=20,
            cmap="RdBu_r",
            alpha=0.8,
        )
        plt.colorbar(label="Predicted Probability (Sum > 0.5)")

        # # Plot the true decision boundary for this slice
        # true_boundary_x0 = x_range.numpy()
        # true_boundary_x1 = 0.5 - true_boundary_x0
        # plt.plot(true_boundary_x0, true_boundary_x1, 'k--', linewidth=3, label='True Boundary (x₁ = 0.5 - x₀)')

        plt.title("Decision Boundary Slice (x₂=0, x₃=0)")
        plt.xlabel("Dimension 0")
        plt.ylabel("Dimension 1")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        mo.output.append(fig)
        mo.output.append(fig2)
        mo.output.append(fig3)
        return

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # What you are seeing.
    1. The left norm, the random variable x for the asymmetric data, starts taking over the majority of the distribution with the right norm still picking up the faint symmetric data. This shows that our model is robust in its symmetric predictions, its even able to detect small symmetry from asymmetry.

    Ok what if we really try to break it, let's think of the most random motion, quantum motion. We will use brownian motion apprioxmiation for the x
    """
    )
    return


@app.cell
def _(ChiralNet, mo, nn, plt, torch):
    def _():
        import numpy as np

        # --- Configuration ---
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def get_4d_chiral_data(n=4096):
            """Generates data for the 4D 'Twist' task where f(x) = f(-x)."""
            x = torch.randn(n, 4, device=DEVICE) * 2
            twist = x[:, 0] * x[:, 3] - x[:, 1] * x[:, 2]
            y = (twist > 0.5).float().unsqueeze(1)
            return x, y

        def brownian_motion(n_steps, dim=4):
            """Simulates a random walk path."""
            path = torch.zeros(n_steps + 1, dim, device=DEVICE)
            steps = torch.randn(n_steps, dim, device=DEVICE) * np.sqrt(
                0.1
            )  # Use sqrt of step size
            path[1:] = torch.cumsum(steps, dim=0)
            return path

        # --- Step 1: Train the "Sensor" Model on the Twist Task ---
        print("--- Step 1: Training the 4D 'Twist' Sensor Model ---")

        # Instantiate and train the model
        sensor_model = ChiralNet(dim=4).to(DEVICE)
        optimizer = torch.optim.AdamW(sensor_model.parameters(), lr=0.005)
        loss_fn = nn.BCEWithLogitsLoss()
        x_train, y_train = get_4d_chiral_data()

        # Short training loop to make the model an expert
        for epoch in range(500):
            sensor_model.train()
            optimizer.zero_grad()
            output, _, _ = sensor_model(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(
                    f"Sensor training | Epoch {epoch + 1:3d}, Loss: {loss.item():.6f}"
                )

        # --- Step 2: Use the Trained Sensor on a Brownian Motion Path ---
        print("\n--- Step 2: Applying Trained Sensor to a New Brownian Path ---")

        # Load the trained model for inference
        sensor_model_loaded = ChiralNet(dim=4).to(DEVICE)
        sensor_model_loaded.eval()

        # Generate a new Brownian motion path
        # Note: 50,000 steps is manageable for most systems. 1M+ can be very slow to plot.
        N_STEPS = 50000
        motion_path = brownian_motion(n_steps=N_STEPS, dim=4)

        # Get model predictions for every point on the path
        with torch.no_grad():
            motion_output, _, _ = sensor_model_loaded(motion_path)
            # Convert logits to probabilities [0, 1] and move to CPU for plotting
            motion_probs = torch.sigmoid(motion_output).squeeze().cpu().numpy()

        # --- Step 3: Visualize the Results ---
        print("--- Step 3: Visualizing the Analyzed Path ---")

        # Move the path data to CPU for numpy/matplotlib
        motion_path_np = motion_path.cpu().numpy()

        # Plot the 3D projection of the 4D path
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Color the path using a colormap based on the model's prediction probability
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        # For performance, we plot using a scatter plot, which is faster for many points
        scatter = ax.scatter(
            motion_path_np[:, 0],
            motion_path_np[:, 1],
            motion_path_np[:, 2],
            c=motion_probs,
            cmap=cmap,
            norm=norm,
            s=2,
            alpha=0.7,
        )

        # Add a colorbar to explain the coloring
        fig.colorbar(
            scatter,
            ax=ax,
            shrink=0.6,
            label='Predicted Probability (Inside "Twist" Region)',
        )

        # Highlight the start and end points
        ax.scatter(
            motion_path_np[0, 0],
            motion_path_np[0, 1],
            motion_path_np[0, 2],
            color="red",
            s=150,
            label="Start",
            depthshade=False,
            edgecolors="w",
        )
        ax.scatter(
            motion_path_np[-1, 0],
            motion_path_np[-1, 1],
            motion_path_np[-1, 2],
            color="blue",
            s=200,
            label="End",
            marker="X",
            depthshade=False,
        )

        ax.set_title(
            "Brownian Motion Path in 3D (Projection), Colored by Model Prediction"
        )
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        ax.set_zlabel("Dimension 2")
        ax.legend()
        plt.tight_layout()

        # Also, plot the probability over the course of the walk
        fig2 = plt.figure(figsize=(14, 5))
        plt.plot(motion_probs, linewidth=1)
        plt.title("Model's Prediction Probability Along the Path's Steps")
        plt.xlabel("Step Number")
        plt.ylabel("Probability (Inside 'Twist' Region)")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.5)
        mo.output.append(fig)
        mo.output.append(fig2)
        return

    _()
    return


if __name__ == "__main__":
    app.run()
