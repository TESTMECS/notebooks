# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
#     "torch==2.7.1",
# ]
# ///

import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    return F, mo, nn, np, plt, torch


@app.cell
def _(torch):
    # --- Configuration ---
    torch.manual_seed(42)

    # Set up the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Brownian Motion Drift Detection Task ---")
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(F, torch):
    # --- Improved Data Generation: Create paths as data points ---
    def generate_brownian_paths(
        n_paths, n_steps, dim, drift_strength=0.1, device="cpu"
    ):
        """
        Generates a batch of Brownian paths. Half are pure, half have hidden drift.
        Each path is flattened into a single vector to be used as input.

        Performance improvements:
        - Direct tensor creation on device
        - More efficient drift application
        - Better random seed handling
        """
        # Ensure even number of paths for clean split
        if n_paths % 2 != 0:
            n_paths += 1

        half_paths = n_paths // 2

        # Create labels: first half is pure (0), second half has drift (1)
        labels = torch.zeros(n_paths, 1, device=device)
        labels[half_paths:] = 1

        # Generate all random steps at once for efficiency, directly on device
        steps = torch.randn(n_paths, n_steps, dim, device=device)

        # --- Create the drift paths more efficiently ---
        # Create a random drift vector for each path in the second half
        drift_vectors = torch.randn(half_paths, dim, device=device)
        # Normalize and scale the drift vectors
        drift_vectors = F.normalize(drift_vectors, p=2, dim=1) * drift_strength

        # More diverse drift patterns - some constant, some varying
        drift_pattern = torch.rand(half_paths, 1, device=device)

        # Apply drift: some paths get constant drift, others get time-varying drift
        for i in range(half_paths):
            if drift_pattern[i] > 0.5:
                # Constant drift
                steps[half_paths + i] += drift_vectors[i].unsqueeze(0)
            else:
                # Time-varying drift (increases over time)
                time_weights = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)
                steps[half_paths + i] += drift_vectors[i].unsqueeze(0) * time_weights

        # Calculate the cumulative sum to get the paths
        paths = torch.cumsum(steps, dim=1)

        # Flatten each path into a single feature vector
        # Shape: [n_paths, n_steps * dim]
        paths_flattened = paths.view(n_paths, -1)

        # Shuffle the data and labels together
        idx = torch.randperm(n_paths, device=device)
        return paths_flattened[idx], labels[idx]

    return (generate_brownian_paths,)


@app.cell
def _(F, nn):
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
def _(mo):
    # --- Training Setup ---
    N_STEPS = 1000
    DIM = 3  
    INPUT_DIM = N_STEPS * DIM  # Flattened input dimension
    epochs = 400  
    lr = 1e-3 
    batch_size = 512  # Add batching for better performance
    DRIFT_STRENGTH = 0.5
    mo.output.append(f"DRIFT_STRENGTH: {DRIFT_STRENGTH}")
    return DIM, DRIFT_STRENGTH, INPUT_DIM, N_STEPS, batch_size, epochs, lr


@app.cell
def _(
    DIM,
    DRIFT_STRENGTH,
    N_STEPS,
    batch_size,
    device,
    generate_brownian_paths,
    torch,
):
    # Generate training and test data directly on device for better performance
    x_train, y_train = generate_brownian_paths(
        n_paths=8192, n_steps=N_STEPS, dim=DIM, device=device
    )
    x_test, y_test = generate_brownian_paths(
        n_paths=2048, n_steps=N_STEPS, dim=DIM, drift_strength=DRIFT_STRENGTH, device=device
    )
    # Create data loaders for better memory efficiency
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader, x_test, y_test


@app.cell
def _(
    ChiralNet,
    INPUT_DIM,
    batch_size,
    device,
    epochs,
    lr,
    mo,
    nn,
    torch,
    train_loader,
    x_test,
    y_test,
):
    # --- Improved Training Loop ---
    # Initialize model and move to device
    model = ChiralNet(dim=INPUT_DIM)
    model = model.to(device)  # Fix: actually move model to device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.8
    )
    loss_fn = nn.BCEWithLogitsLoss()
    left_norms, right_norms, losses = [], [], []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_l_norm = 0
        epoch_r_norm = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output, l_norm, r_norm = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_l_norm += l_norm.item()
            epoch_r_norm += r_norm.item()
            num_batches += 1

        # Average the metrics over all batches
        avg_loss = epoch_loss / num_batches
        avg_l_norm = epoch_l_norm / num_batches
        avg_r_norm = epoch_r_norm / num_batches

        scheduler.step(avg_loss)

        if epoch % 25 == 0:  # Record more frequently
            left_norms.append(avg_l_norm)
            right_norms.append(avg_r_norm)
            losses.append(avg_loss)

            if epoch % 200 == 0:
                mo.output.append(
                    f"Epoch {epoch:5d} | Loss: {avg_loss:.6f} | L Norm: {avg_l_norm:.3f} | R Norm: {avg_r_norm:.3f}"
                )

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        # Evaluate in batches to avoid memory issues
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        all_outputs = []
        for batch_x, _ in test_loader:
            output, _, _ = model(batch_x)
            all_outputs.append(output)

        output = torch.cat(all_outputs, dim=0)
        accuracy = ((torch.sigmoid(output) > 0.5) == y_test).float().mean()

    mo.output.append(f"\nFinal Test Accuracy: {accuracy.item():.4f}")
    return left_norms, losses, output, right_norms


@app.cell
def _(
    DIM,
    N_STEPS,
    left_norms,
    losses,
    mo,
    np,
    output,
    plt,
    right_norms,
    torch,
    x_test,
    y_test,
):
    # --- Visualization ---
    # 1. Plot Chiral Dominance - This is the key result!
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Chiral Dominance
    plt.subplot(1, 2, 1)
    epochs_recorded = [i * 25 for i in range(len(left_norms))]
    plt.plot(epochs_recorded, left_norms, label="Left Path Norm", linewidth=2)
    plt.plot(epochs_recorded, right_norms, label="Right Path Norm", linewidth=2)
    plt.title("Chiral Dominance Evolution")
    plt.xlabel("Training Epochs")
    plt.ylabel("Pathway Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)


    # Plot 2: Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_recorded, losses, label="Training Loss", color="red", linewidth=2)
    plt.title("Training Loss Evolution")
    plt.xlabel("Training Epochs")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.output.append(fig)

    # 2. Visualize some example paths from the test set
    fig = plt.figure(figsize=(12, 10))
    y_test_np = y_test.cpu().numpy().squeeze()
    preds_np = (torch.sigmoid(output).cpu().numpy() > 0.5).squeeze()

    # Plot a correct "No Drift" classification
    ax1 = fig.add_subplot(221, projection="3d")
    path_idx = np.where((y_test_np == 0) & (preds_np == 0))[0][0]
    path = x_test[path_idx].cpu().view(N_STEPS, DIM).numpy()
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], color="blue")
    ax1.set_title("Correctly Classified: Pure Brownian")

    # Plot a correct "Drift" classification
    ax2 = fig.add_subplot(222, projection="3d")
    path_idx = np.where((y_test_np == 1) & (preds_np == 1))[0][0]
    path = x_test[path_idx].cpu().view(N_STEPS, DIM).numpy()
    ax2.plot(path[:, 0], path[:, 1], path[:, 2], color="green")
    ax2.set_title("Correctly Classified: Hidden Drift")

    # Plot an incorrect classification (if any)
    ax3 = fig.add_subplot(223, projection="3d")
    incorrect_indices = np.where(y_test_np != preds_np)[0]
    if len(incorrect_indices) > 0:
        path_idx = incorrect_indices[0]
        path = x_test[path_idx].cpu().view(N_STEPS, DIM).numpy()
        ax3.plot(path[:, 0], path[:, 1], path[:, 2], color="red")
        ax3.set_title(
            f"Misclassified (True: {int(y_test_np[path_idx])}) - Total Misclassifications: {len(incorrect_indices)}"
        )
    else:
        ax3.set_title("No Misclassifications Found")

    plt.tight_layout()
    mo.output.append(fig)
    return preds_np, y_test_np


@app.cell
def _(mo, np, preds_np, y_test_np):
    # Calculate the number of correctly and incorrectly classified samples
    num_correct = np.sum(y_test_np == preds_np)
    num_incorrect = np.sum(y_test_np != preds_np)

    mo.output.append(f"Total test samples: {len(y_test_np)}")
    mo.output.append(f"Number of correctly classified samples: {num_correct}")
    mo.output.append(f"Number of misclassified samples: {num_incorrect}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What This Notebook Demonstrates

    ### 🎯 **Core Concept: Chiral Neural Networks for Brownian Motion Analysis**

    This notebook implements and demonstrates a **Chiral Neural Network** applied to detect hidden drift in Brownian motion paths. Here's what's happening:

    ### 📊 **The Task: Brownian Motion Drift Detection**
    - **Input**: 3D Brownian motion paths (sequences of random walk steps)
    - **Challenge**: Classify whether a path contains hidden drift or is pure random motion
    - **Difficulty**: The drift is subtle and requires detecting long-term directional bias in seemingly random data

    ### 🧠 **The Chiral Architecture**
    The neural network uses a "chiral" (handedness) design with two pathways:
    - **Left Pathway**: Processes the original path data
    - **Right Pathway**: Processes the mirror image (negative) of the path data
    - **Competition**: The final decision comes from `Left - Right` features

    ### 🔍 **Key Observations**

    #### **1. Chiral Dominance Plot**
    The first visualization shows how the left and right pathway norms evolve during training:
    - **Symmetry Breaking**: Initially both pathways are similar, but one becomes dominant
    - **Specialization**: The dominant pathway learns to detect the subtle drift patterns
    - **Emergent Chirality**: The network spontaneously develops "handedness" to solve the task

    #### **2. Path Visualizations**
    The 3D plots show:
    - **Blue paths**: Pure Brownian motion (no drift) - correctly classified
    - **Green paths**: Hidden drift paths - correctly classified  
    - **Red paths**: Misclassified examples (if any)

    ### 📈 Expected Results

    - Symmetry Baseline: With no hidden drift, the two chiral pathways (processing x and -x) should respond similarly. Thus, the model's decision boundary is effectively random, leading to ~0.5 test accuracy.

    - Symmetry Breaking via Drift: As the drift strength increases, the symmetry is disrupted. One pathway (e.g., the left) increasingly aligns with the directional bias, while the other diverges. This is reflected in the divergence of their activation norms.

    - Emergent Differentiation: The model isn't explicitly given a difference vector—it learns to compute one via competing activations. This resembles a form of automatic differentiation between symmetric and asymmetric components of stochastic motion.

    - Dominance Signal: One pathway should become consistently stronger, acting as a detector for hidden structure.

    - Loss Behavior: A well-designed architecture should yield a smooth, stable decline in loss, especially as it starts capturing the directional signal more effectively.

    This demonstrates how bio-inspired asymmetry (chirality) enables the network to detect subtle, directional structure in random data—something symmetric architectures may overlook.
    """
    )
    return


if __name__ == "__main__":
    app.run()
