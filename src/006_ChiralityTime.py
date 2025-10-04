# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
#     "rich==14.0.0",
#     "scikit-learn==1.7.0",
#     "torch==2.7.1",
# ]
# ///

import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # ⏰ The Arrow of Time Detector with Enhanced ChiralNet

    ## 🎯 Experimental Design: The "Causality Witness"

    ### The "Objects" of Our System

    We work with two fundamentally different types of time-series data:

    **🔄 Reversible Process (Label 0)**: A process that is statistically identical when played forwards or backwards. The Ornstein-Uhlenbeck (OU) process is our perfect candidate - a "mean-reverting" random walk, like a particle in a bowl of molasses being jostled by random forces. It wiggles around an average value but has no overall trend. Its future and past look identical.

    **➡️ Irreversible Process (Label 1)**: A process with a clear, undeniable "arrow." A random walk with strong, consistent drift - like a particle being pushed by constant wind. Played in reverse, it looks deeply unnatural, as if the particle is being "sucked" towards its origin against the random forces.

    ### 🧠 The ChiralNet Setup
    - **Architecture**: 1D Convolutional Neural Network (CNN) optimized for sequential pattern detection
    - **Left Pathway**: Sees time series in chronological order (t₁ → t₂ → t₃)  
    - **Right Pathway**: Sees time-reversed sequence (t₃ → t₂ → t₁)
    - **Task**: "Is this an irreversible process?"

    ### 💡 Hypothesis
    When the network encounters:
    - **Reversible (OU) process**: Forward and backward views contain identical statistical information → Symmetric input
    - **Irreversible (drift) process**: Forward and backward views are fundamentally different → Asymmetric input

    **Prediction**: The network cannot simply ignore one pathway. It will be forced into cooperation, with both pathway norms rising together as the ChiralNet learns that the relationship between forward and backward views is the key to identifying the direction of time even in reversible processes.
    """
    )
    return


@app.cell
def _():
    # --- The Arrow of Time Detector with ChiralNet ---
    import logging

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from rich.console import Console
    from rich.logging import RichHandler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    # Setup Rich logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger("chirality_time")
    console = Console()

    return (
        DataLoader,
        StandardScaler,
        TensorDataset,
        logger,
        logging,
        mo,
        nn,
        np,
        plt,
        torch,
        train_test_split,
    )


@app.cell
def _(logging, torch):
    # --- Configuration ---
    torch.manual_seed(42)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    # --- Time Series Configuration ---
    SEQUENCE_LENGTH = 100
    N_SAMPLES = 8192
    logging.info("--- Arrow of Time Detector ---")
    logging.info(f"Using device: {DEVICE}")
    return (
        BATCH_SIZE,
        DEVICE,
        EPOCHS,
        LEARNING_RATE,
        N_SAMPLES,
        SEQUENCE_LENGTH,
    )


@app.cell
def _(nn, torch):
    # --- Data Generation ---
    def generate_time_series_data(n_samples, seq_len):
        """
        Generates two types of time series:
        1. Reversible (Ornstein-Uhlenbeck process, label 0)
        2. Irreversible (Random walk with drift, label 1)
        """
        sequences = torch.zeros(n_samples, seq_len)
        labels = torch.zeros(n_samples, 1)

        for i in range(n_samples):
            # 50% chance for each type of process
            if i % 2 == 0:
                # Reversible: Ornstein-Uhlenbeck process (mean-reverting)
                # x_t = x_{t-1} + theta * (mu - x_{t-1}) + noise
                theta = 0.1  # Strength of mean reversion
                mu = 0.0  # Mean to revert to
                sigma = 0.2  # Volatility
                path = torch.zeros(seq_len)
                for t in range(1, seq_len):
                    path[t] = (
                        path[t - 1]
                        + theta * (mu - path[t - 1])
                        + sigma * torch.randn(1)
                    )
                sequences[i] = path
                labels[i] = 0.0  # Reversible
            else:
                # Irreversible: Random walk with a strong drift
                drift = 0.5
                sigma = 0.5
                steps = torch.randn(seq_len) * sigma + drift
                sequences[i] = torch.cumsum(steps, dim=0)
                labels[i] = 1.0  # Irreversible

        return sequences.unsqueeze(1), labels  # Add channel dim for CNN

    # --- Model Definition ---
    class ChiralTimeNet(nn.Module):
        """A Chiral Differential Engine using a CNN to detect temporal asymmetry."""

        def __init__(self, in_channels=1, final_feature_dim=64):
            super().__init__()
            self.cnn_pathway = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, final_feature_dim),
                nn.ReLU(),
            )
            self.output_layer = nn.Linear(final_feature_dim, 1)

        def forward(self, x_forward, x_reversed):
            # Inputs already have channel dimension
            l_out = self.cnn_pathway(x_forward)
            r_out = self.cnn_pathway(x_reversed)

            net_difference = l_out - r_out
            final_output = self.output_layer(net_difference)
            return final_output, l_out.norm(), r_out.norm()

    return ChiralTimeNet, generate_time_series_data


@app.cell
def _(
    BATCH_SIZE,
    ChiralTimeNet,
    DEVICE,
    DataLoader,
    EPOCHS,
    LEARNING_RATE,
    N_SAMPLES,
    SEQUENCE_LENGTH,
    TensorDataset,
    generate_time_series_data,
    logging,
    nn,
    np,
    torch,
    train_test_split,
):
    # --- Training Setup ---
    x_data, y_data = generate_time_series_data(N_SAMPLES, SEQUENCE_LENGTH)

    # Split data
    indices = np.arange(N_SAMPLES)
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_data
    )

    x_train, y_train = x_data[train_indices], y_data[train_indices]
    x_test, y_test = x_data[test_indices], y_data[test_indices]


    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )

    model = ChiralTimeNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # --- Training & Evaluation ---
    logging.info("\n--- Training ChiralNet as a Causality Witness ---")
    left_norms, right_norms, losses = [], [], []
    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch in train_loader:
            x_forward = x_batch.to(DEVICE)
            # Create the time-reversed version on the fly
            x_reversed = torch.flip(x_forward, [2])
            labels = y_batch.to(DEVICE)

            optimizer.zero_grad()
            output, l_norm, r_norm = model(x_forward, x_reversed)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        left_norms.append(l_norm.item())
        right_norms.append(r_norm.item())
        if (epoch + 1) % 2 == 0:
            logging.info(f"Epoch {epoch + 1:2d} | Loss: {loss.item():.6f}")

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        x_forward_test = x_test.to(DEVICE)
        x_reversed_test = torch.flip(x_forward_test, [2])
        test_output, _, _ = model(x_forward_test, x_reversed_test)
        accuracy = (
            ((torch.sigmoid(test_output) > 0.5) == y_test.to(DEVICE))
            .float()
            .mean()
            .item()
        )
    logging.info(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    return left_norms, losses, right_norms, x_test, y_test


@app.cell
def _(left_norms, losses, mo, np, plt, right_norms, x_test, y_test):
    # --- Visualization ---
    fig2 = plt.figure(figsize=(14, 12))

    # Plot Pathway Norms
    plt.subplot(2, 1, 1)
    plt.plot(left_norms, label="Path Norm (Forward Time)")
    plt.plot(right_norms, label="Path Norm (Reversed Time)")
    plt.title("Pathway Norms on 'Arrow of Time' Task")
    plt.xlabel("Epoch")
    plt.ylabel("Pathway Norm")
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)

    # Visualize example time series
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Find one example of each class from the test set
    reversible_idx = np.where(y_test.numpy() == 0)[0][0]
    irreversible_idx = np.where(y_test.numpy() == 1)[0][0]
    ax1.plot(x_test[reversible_idx].squeeze().numpy())
    ax1.set_title("Example of a Reversible Process (OU)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.grid(True)
    ax2.plot(x_test[irreversible_idx].squeeze().numpy())
    ax2.set_title("Example of an Irreversible Process (Drift)")
    ax2.set_xlabel("Time")
    ax2.grid(True)

    plt.tight_layout()
    mo.output.append(fig)
    mo.output.append(fig2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Findings:
    - Chiral Net learned to use both path norms to predict the label (0 for Reversible and 1 for Irreversible)
    - In, this way we show that the Chiral Network learns features of both non-reversible and reversible networks and is able to predict the label of both.
    - **Synthetic data gap** — The generated data (OU vs. drift) may be too clean and separable, unlike real-world cases where forward and backward might only differ subtly.

    ## So what's the point then?
    This experiment is proof-of-concept: if a network can distinguish forward from backward time on idealized physical processes, it hints that time asymmetry is detectable from local statistics in dynamics. It's like teaching a machine what entropy feels like without needing equations.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Now we are going to try something harder. A Henon Map is a map of the attractor of the Henon
    map which is a special case of the Lorenz map. Basically its a discrete but chaotic
    attractor.
    """
    )
    return


@app.cell
def _(StandardScaler, logger, mo, nn, np, plt, torch, train_test_split):
    def _():

        # Optimized Henon Map System (Discrete Chaotic Map)
        def henon_map(x, y, a=1.4, b=0.3, max_val=100.0):
            """
            Henon map - discrete chaotic system with numerical stability
            Much faster than ODE systems since no integration required
            """
            # Clip values to prevent overflow
            x = np.clip(x, -max_val, max_val)
            y = np.clip(y, -max_val, max_val)

            x_new = 1 - a * x**2 + y
            y_new = b * x

            # Additional clipping after computation
            x_new = np.clip(x_new, -max_val, max_val)
            y_new = np.clip(y_new, -max_val, max_val)

            return x_new, y_new

        def generate_henon_data(num_steps, num_trajectories=12, noise_level=1e-6):
            """
            Generate Henon map data - very fast discrete iteration
            Higher trajectory count since it's computationally cheap
            """
            all_trajectories = []

            logger.info(f"🌀 Generating {num_trajectories} Henon map trajectories...")

            for i in range(num_trajectories):
                # More conservative initial conditions to prevent overflow
                x0 = np.random.uniform(-1.5, 1.5)
                y0 = np.random.uniform(-1.5, 1.5)

                # Use classic parameters for stability
                a = np.random.uniform(1.38, 1.42)  # Very close to classic value 1.4
                b = np.random.uniform(0.28, 0.32)  # Very close to classic value 0.3

                # Fast iteration with overflow protection
                trajectory = np.zeros((num_steps, 2))
                x, y = x0, y0

                valid_steps = 0
                for t in range(num_steps):
                    # Check for numerical issues
                    if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                        logger.warning(
                            f"  ⚠️ Numerical instability in trajectory {i + 1} at step {t}, restarting..."
                        )
                        # Restart with new initial conditions
                        x = np.random.uniform(-1.0, 1.0)
                        y = np.random.uniform(-1.0, 1.0)

                    trajectory[t] = [x, y]
                    x, y = henon_map(x, y, a, b)

                    # Add tiny noise for numerical stability
                    x += np.random.normal(0, noise_level)
                    y += np.random.normal(0, noise_level)

                    valid_steps += 1

                    # Early termination if trajectory escapes to infinity
                    if abs(x) > 50 or abs(y) > 50:
                        logger.warning(
                            f"  ⚠️ Trajectory {i + 1} escaped bounds at step {t}, truncating..."
                        )
                        break

                # Only use the valid portion of the trajectory
                if (
                    valid_steps > num_steps // 2
                ):  # At least half the trajectory should be valid
                    all_trajectories.append(trajectory[:valid_steps])
                    if (i + 1) % 3 == 0:
                        logger.info(
                            f"  ⚡ Completed {i + 1}/{num_trajectories} trajectories ({valid_steps} valid steps)"
                        )
                else:
                    logger.warning(
                        f"  ❌ Trajectory {i + 1} had too few valid steps ({valid_steps}), skipping..."
                    )

            if len(all_trajectories) == 0:
                raise ValueError(
                    "No valid Henon trajectories generated! Try different parameters."
                )

            # Concatenate and normalize
            full_data = np.vstack(all_trajectories)
            logger.info(f"  ✅ Total valid data points: {full_data.shape[0]}")

            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(full_data)

            return normalized_data, scaler

        # Reuse the enhanced time series function from Lorenz experiment
        def create_enhanced_time_series_pairs(data, sequence_length, stride=1):
            """Enhanced version with stride control"""
            sequences = []
            labels = []

            # Extract overlapping subsequences with specified stride
            for i in range(0, len(data) - sequence_length + 1, stride):
                subseq = data[i : i + sequence_length]

                # Forward sequence (label = 1)
                sequences.append(subseq)
                labels.append(1.0)

                # Reversed sequence (label = 0)
                reversed_subseq = np.flip(subseq, axis=0)
                sequences.append(reversed_subseq)
                labels.append(0.0)

            sequences = np.array(sequences)
            labels = np.array(labels)

            # Convert to tensors
            X = torch.tensor(sequences, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

            # Create pairs: (original, reversed_original)
            X_forward = X[::2]  # Every even index (forward sequences)
            X_reversed = X[1::2]  # Every odd index (reversed sequences)
            y_labels = y[::2]  # Labels for forward sequences

            # Combine and shuffle
            combined_X1 = torch.cat([X_forward, X_reversed], dim=0)
            combined_X2 = torch.cat([X_reversed, X_forward], dim=0)
            combined_y = torch.cat([y_labels, 1.0 - y_labels], dim=0)

            # Shuffle
            perm = torch.randperm(len(combined_X1))
            return combined_X1[perm], combined_X2[perm], combined_y[perm]

        # Enhanced ChiralNet architecture
        class ImprovedTimeSeriesChiralNet(nn.Module):
            def __init__(self, input_dim, sequence_length, hidden_dim=128):
                super().__init__()
                self.input_dim = input_dim
                self.sequence_length = sequence_length

                # Multi-scale CNN pathway
                self.feature_extractor = nn.Sequential(
                    nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(64, 96, kernel_size=5, padding=2),
                    nn.BatchNorm1d(96),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(96, hidden_dim, kernel_size=7, padding=3),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )

                # Enhanced output layers
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 4, 1),
                )

            def forward(self, x1, x2):
                # x1: forward time series, x2: reversed time series
                x1 = x1.transpose(1, 2)  # [batch, input_dim, sequence_length]
                x2 = x2.transpose(1, 2)

                # Extract features from both pathways
                features1 = self.feature_extractor(x1).squeeze(
                    -1
                )  # [batch, hidden_dim]
                features2 = self.feature_extractor(x2).squeeze(-1)

                # Chiral difference - the key insight
                chiral_difference = features1 - features2

                # Classification based on chiral asymmetry
                output = self.classifier(chiral_difference)

                return output, features1, features2

        def run_henon_experiment():
            logger.info("🌀 Starting Henon Map Arrow of Time Experiment")

            # Optimized parameters - can afford more data since Henon is fast
            num_steps = 50000  # More steps since iteration is fast
            num_trajectories = 12  # More trajectories since computation is cheap
            sequence_length = 100  # Longer sequences since we have more data
            stride = 20  # Larger stride for diversity

            logger.info(
                f"⚡ Parameters: {num_steps} steps, {num_trajectories} trajectories, seq_len={sequence_length}"
            )

            # Generate data - very fast for discrete maps
            logger.info("🌀 Generating Henon map chaotic dynamics...")
            data, scaler = generate_henon_data(num_steps, num_trajectories)
            logger.info(
                f"✅ Generated {data.shape[0]} time steps with {data.shape[1]} dimensions"
            )

            # Create sequence pairs
            logger.info("🔄 Creating sequence pairs...")
            X1, X2, y = create_enhanced_time_series_pairs(data, sequence_length, stride)
            logger.info(f"✅ Created {len(X1)} sequence pairs")

            # Train/test split
            X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
                X1, X2, y, test_size=0.2, random_state=42, stratify=y
            )

            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"💾 Using device: {device}")

            X1_train = X1_train.to(device)
            X2_train = X2_train.to(device)
            y_train = y_train.to(device)
            X1_test = X1_test.to(device)
            X2_test = X2_test.to(device)
            y_test = y_test.to(device)

            # Model setup - 2D input for Henon map
            model = ImprovedTimeSeriesChiralNet(
                input_dim=2, sequence_length=sequence_length, hidden_dim=64
            ).to(device)

            # Optimized training
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.004, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.008, epochs=120, steps_per_epoch=len(X1_train) // 64
            )
            loss_fn = nn.BCEWithLogitsLoss()

            # Training setup
            batch_size = 64
            epochs = 120

            logger.info("🎯 Starting Henon map training...")
            train_losses = []
            train_accuracies = []
            val_accuracies = []
            l_norms = []
            r_norms = []

            train_dataset = torch.utils.data.TensorDataset(X1_train, X2_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )

            best_val_acc = 0.0
            patience = 20
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                for batch_x1, batch_x2, batch_y in train_loader:
                    optimizer.zero_grad()

                    output, feat1, feat2 = model(batch_x1, batch_x2)
                    loss = loss_fn(output, batch_y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    predicted = (torch.sigmoid(output) > 0.5).float()
                    epoch_correct += (predicted == batch_y).sum().item()
                    epoch_total += batch_y.size(0)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_output, val_feat1, val_feat2 = model(X1_test, X2_test)
                    val_predicted = (torch.sigmoid(val_output) > 0.5).float()
                    val_accuracy = (val_predicted == y_test).float().mean().item()

                # Track metrics
                avg_loss = epoch_loss / len(train_loader)
                train_accuracy = epoch_correct / epoch_total

                train_losses.append(avg_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                l_norms.append(val_feat1.norm().item())
                r_norms.append(val_feat2.norm().item())

                # Early stopping
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
                    break

                # Progress reporting
                if (epoch + 1) % 25 == 0 or epoch == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"📈 Epoch {epoch + 1:3d}/{epochs} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"Train Acc: {train_accuracy:.4f} | "
                        f"Val Acc: {val_accuracy:.4f} | "
                        f"LR: {lr:.6f}"
                    )

            # Final evaluation
            model.eval()
            with torch.no_grad():
                final_output, _, _ = model(X1_test, X2_test)
                final_predicted = (torch.sigmoid(final_output) > 0.5).float()
                final_accuracy = (final_predicted == y_test).float().mean().item()
                final_loss = loss_fn(final_output, y_test).item()

            logger.info("🎯 Final Results Computed")

            # Performance analysis
            if final_accuracy > 0.8:
                performance_emoji = "🎉"
                performance_text = "EXCELLENT"
                performance_desc = "Strong discrete chaos detection!"
            elif final_accuracy > 0.7:
                performance_emoji = "✅"
                performance_text = "GOOD"
                performance_desc = "Clear map asymmetry detection!"
            elif final_accuracy > 0.6:
                performance_emoji = "⚠️"
                performance_text = "MODERATE"
                performance_desc = "Some discrete pattern recognition"
            else:
                performance_emoji = "❌"
                performance_text = "POOR"
                performance_desc = "Weak temporal detection"

            logger.info(f"{performance_emoji} {performance_text}: {performance_desc}")

            # Visualization
            logger.info("📊 Generating Henon map visualizations...")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Training curves
            axes[0, 0].plot(
                train_losses, label="Training Loss", alpha=0.8, color="#e74c3c"
            )
            axes[0, 0].set_title(
                "Henon Map: Training Loss", fontsize=14, fontweight="bold"
            )
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("BCE Loss")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            axes[0, 1].plot(
                train_accuracies, label="Training Accuracy", alpha=0.8, color="#3498db"
            )
            axes[0, 1].plot(
                val_accuracies, label="Validation Accuracy", alpha=0.8, color="#2ecc71"
            )
            axes[0, 1].set_title("Henon Map: Accuracy", fontsize=14, fontweight="bold")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

            # Pathway norms
            axes[1, 0].plot(
                l_norms, label="Forward Pathway Norm", alpha=0.8, color="#9b59b6"
            )
            axes[1, 0].plot(
                r_norms, label="Reversed Pathway Norm", alpha=0.8, color="#f39c12"
            )
            axes[1, 0].set_title(
                "Henon Map: Pathway Norms", fontsize=14, fontweight="bold"
            )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Feature Norm")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

            # Henon map attractor
            sample_data = data[:5000]  # More points to show attractor structure
            axes[1, 1].scatter(
                sample_data[:, 0], sample_data[:, 1], alpha=0.6, s=0.5, color="#8e44ad"
            )
            axes[1, 1].set_title(
                "Henon Map: Strange Attractor", fontsize=14, fontweight="bold"
            )
            axes[1, 1].set_xlabel("x (normalized)")
            axes[1, 1].set_ylabel("y (normalized)")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            mo.output.append(fig)

            # Performance comparison plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.bar(
                ["Training", "Validation", "Test"],
                [train_accuracies[-1], best_val_acc, final_accuracy],
                color=["#3498db", "#2ecc71", "#e74c3c"],
                alpha=0.8,
            )
            ax.set_title(
                "Henon Map: Performance Comparison", fontsize=16, fontweight="bold"
            )
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            for i, v in enumerate([train_accuracies[-1], best_val_acc, final_accuracy]):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")


            mo.output.append(fig)

            logger.info("✅ Henon Map experiment completed")
            return 

        # Run the Henon map experiment
        return run_henon_experiment()

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    # 🔍 ChiralNet: Can a Neural Net Sense the Flow of Time?

    This experiment explores a deep question:  
    **Can we teach a machine to tell if a signal is running forward or backward in time?**

    ### 🧪 What We Did

    We built a special neural network called **ChiralNet** and gave it two types of signals:
    1. **Reversible signals** – like a random wiggle that looks the same forward or backward.
    2. **Irreversible signals** – like a process being pushed in one direction (like wind or drift). Playing it backward just looks *wrong*.

    We trained ChiralNet to answer one simple question:  
    > "Does this signal have a clear arrow of time?"

    To test it harder, we gave it signals from a **chaotic system** called the **Henon map** — a simple math formula that produces wild, unpredictable motion. It's fast, efficient, and captures the essence of chaos without using differential equations.

    ### 📊 Summary

    | System      | Type       | What Happened                     |
    |-------------|------------|-----------------------------------|
    | Drift Walk  | Simple     | Time direction was super obvious |
    | OU Process  | Reversible | Harder, but ChiralNet figured it out |
    | Henon Map   | Chaotic    | ChiralNet still nailed it        |

    # Future Tests
    - Lorenz attractor (continuous ODE)
    - Double pendulum
    - Real-world time series (motion sensor, EEG, audio, finance)
    """
    )
    return


if __name__ == "__main__":
    app.run()
