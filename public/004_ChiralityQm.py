# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
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
    # 📖 Abstract

    ChiralNet, a neural network architecture designed to act as a learned quantum entanglement witness. By comparing a full two-qubit quantum state ρ with its separated product form ρ_A ⊗ ρ_B, ChiralNet learns to detect the "chirality" or asymmetry characteristic of entangled states. Unlike traditional entanglement criteria (e.g., PPT or CHSH), which are often limited to specific cases or assumptions, ChiralNet operates as a general-purpose detector and regressor, trained on simulated quantum states. Demonstrating its effectiveness in both binary classification (entangled vs. separable) and continuous regression (predicting concurrence) tasks, achieving high accuracy and robust generalization under noise. This approach opens the door to learned entanglement witnesses that scale to more complex or noisy quantum systems.
    """
    )
    return


@app.cell
def _():
    # --- Quantum Entanglement Witness with ChiralNet ---
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    import marimo as mo
    return DataLoader, TensorDataset, mo, nn, np, plt, torch, train_test_split


@app.cell
def _(torch):
    # --- Configuration ---
    torch.manual_seed(42)
    EPOCHS = 1000
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Toy Quantum System Configuration ---
    # Two-qubit system. Each qubit is 2D, total system is 4D. Density matrix is 4x4.
    SYSTEM_DIM = 4
    SUBSYSTEM_DIM = 2
    print("--- Quantum Entanglement Witness ---")
    print(f"Using device: {DEVICE}")
    return BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, SUBSYSTEM_DIM, SYSTEM_DIM


@app.cell
def _(SUBSYSTEM_DIM, SYSTEM_DIM, np, torch):
    # --- Optimized Quantum Helper Functions ---
    def get_random_density_matrices_batch(batch_size, dim, device=None):
        """Vectorized generation of random valid density matrices."""
        device = device or torch.device("cpu")
        # Create batch of random complex matrices
        mat = torch.randn(batch_size, dim, dim, dtype=torch.cfloat, device=device)
        # Create Hermitian, positive semi-definite matrices in batch
        psd_mat = torch.bmm(mat, mat.conj().transpose(-2, -1))
        # Normalize by trace to ensure Tr(rho) = 1 for each matrix
        traces = (
            torch.diagonal(psd_mat, dim1=-2, dim2=-1)
            .sum(dim=-1, keepdim=True)
            .unsqueeze(-1)
        )
        return psd_mat / traces

    def partial_trace_batch(rho_batch, keep_subsystem=0):
        """Vectorized partial trace calculation for batch of 4x4 density matrices."""
        batch_size = rho_batch.shape[0]
        # Reshape batch of 4x4 matrices into batch of 2x2x2x2 tensors
        rho_tensor = rho_batch.view(batch_size, 2, 2, 2, 2)

        if keep_subsystem == 0:  # Trace out B to keep A
            reduced_rho = torch.einsum("bijik->bjk", rho_tensor)
        else:  # Trace out A to keep B
            reduced_rho = torch.einsum("bjiki->bjk", rho_tensor)
        return reduced_rho

    # --- Optimized Data Generation ---
    def generate_entanglement_data_vectorized(n_samples, device=None):
        """Highly optimized vectorized data generation."""
        device = device or torch.device("cpu")

        print(f"Generating {n_samples} quantum states (vectorized)...")

        # Pre-allocate tensors for better memory efficiency
        half_samples = n_samples // 2

        # Generate separable states (batch operation)
        print("Generating separable states...")
        rho_A_sep = get_random_density_matrices_batch(
            half_samples, SUBSYSTEM_DIM, device
        )
        rho_B_sep = get_random_density_matrices_batch(
            half_samples, SUBSYSTEM_DIM, device
        )

        # Vectorized Kronecker product for separable states
        rho_separable = torch.zeros(
            half_samples, SYSTEM_DIM, SYSTEM_DIM, dtype=torch.cfloat, device=device
        )
        for i in range(SUBSYSTEM_DIM):
            for j in range(SUBSYSTEM_DIM):
                for k in range(SUBSYSTEM_DIM):
                    for l in range(SUBSYSTEM_DIM):
                        rho_separable[
                            :, i * SUBSYSTEM_DIM + j, k * SUBSYSTEM_DIM + l
                        ] = rho_A_sep[:, i, k] * rho_B_sep[:, j, l]

        # Generate entangled states (batch operation)
        print("Generating entangled states...")
        bell_state_vec = (1 / np.sqrt(2)) * torch.tensor(
            [1, 0, 0, 1], dtype=torch.cfloat, device=device
        )
        rho_bell = torch.outer(bell_state_vec, bell_state_vec.conj())

        # Add noise to Bell states in batch
        noise_batch = (
            get_random_density_matrices_batch(half_samples, SYSTEM_DIM, device) * 0.2
        )
        rho_entangled = rho_bell.unsqueeze(0) + noise_batch

        # Renormalize entangled states
        traces = (
            torch.diagonal(rho_entangled, dim1=-2, dim2=-1)
            .sum(dim=-1, keepdim=True)
            .unsqueeze(-1)
        )
        rho_entangled = rho_entangled / traces

        # Calculate separated parts for entangled states
        rho_A_ent = partial_trace_batch(rho_entangled, keep_subsystem=0)
        rho_B_ent = partial_trace_batch(rho_entangled, keep_subsystem=1)

        # Vectorized Kronecker product for entangled separated parts
        rho_ent_separated = torch.zeros(
            half_samples, SYSTEM_DIM, SYSTEM_DIM, dtype=torch.cfloat, device=device
        )
        for i in range(SUBSYSTEM_DIM):
            for j in range(SUBSYSTEM_DIM):
                for k in range(SUBSYSTEM_DIM):
                    for l in range(SUBSYSTEM_DIM):
                        rho_ent_separated[
                            :, i * SUBSYSTEM_DIM + j, k * SUBSYSTEM_DIM + l
                        ] = rho_A_ent[:, i, k] * rho_B_ent[:, j, l]

        # Combine separable and entangled data
        x1_data = torch.cat([rho_separable, rho_entangled], dim=0)
        x2_data = torch.cat([rho_separable, rho_ent_separated], dim=0)

        # Create labels (0 for separable, 1 for entangled)
        y_data = torch.cat(
            [
                torch.zeros(half_samples, 1, device=device),
                torch.ones(half_samples, 1, device=device),
            ],
            dim=0,
        )

        # Convert complex matrices to real/imag channels efficiently
        x1_real = torch.stack([x1_data.real, x1_data.imag], dim=1)
        x2_real = torch.stack([x2_data.real, x2_data.imag], dim=1)

        # Shuffle the data
        perm = torch.randperm(n_samples, device=device)
        return x1_real[perm], x2_real[perm], y_data[perm]
    return (generate_entanglement_data_vectorized,)


@app.cell
def _(nn):
    # --- Optimized Model Architecture ---
    class OptimizedQuantumNet(nn.Module):
        """Optimized neural network specifically designed for 4x4 quantum matrices."""

        def __init__(self, in_channels=2):
            super().__init__()

            # Specialized layers for 4x4 matrices
            self.feature_extractor = nn.Sequential(
                # First conv layer with smaller kernel for 4x4 input
                nn.Conv2d(in_channels, 32, kernel_size=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                # Second conv layer
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Global average pooling instead of flatten for efficiency
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                # Smaller dense layers since input is small
                nn.Linear(64 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
            )

            # Output layer for difference
            self.classifier = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
            )

        def forward(self, x1, x2):
            # Extract features from both inputs
            feat1 = self.feature_extractor(x1)
            feat2 = self.feature_extractor(x2)

            # Compute difference features
            diff_features = feat1 - feat2

            # Classify based on difference
            output = self.classifier(diff_features)

            return output, feat1.norm(dim=1).mean(), feat2.norm(dim=1).mean()
    return (OptimizedQuantumNet,)


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    DataLoader,
    TensorDataset,
    generate_entanglement_data_vectorized,
    torch,
    train_test_split,
):
    # --- Optimized Training Setup ---
    N_SAMPLES = 8192  # Increased for better training

    # Generate data on GPU if available for faster processing
    x1_data, x2_data, y_data = generate_entanglement_data_vectorized(N_SAMPLES, DEVICE)

    # Move to CPU for train/test split if needed
    if DEVICE.type == "cuda":
        x1_cpu, x2_cpu, y_cpu = x1_data.cpu(), x2_data.cpu(), y_data.cpu()
    else:
        x1_cpu, x2_cpu, y_cpu = x1_data, x2_data, y_data

    # Train/test split
    indices = list(range(N_SAMPLES))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_cpu.numpy().flatten()
    )

    x1_train, x2_train, y_train = (
        x1_cpu[train_indices],
        x2_cpu[train_indices],
        y_cpu[train_indices],
    )
    x1_test, x2_test, y_test = (
        x1_cpu[test_indices],
        x2_cpu[test_indices],
        y_cpu[test_indices],
    )

    # Create optimized data loaders
    train_loader = DataLoader(
        TensorDataset(x1_train, x2_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,  # Faster GPU transfer
        num_workers=2 if torch.cuda.is_available() else 0,
    )
    return train_loader, x1_test, x2_test, y_test


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    DataLoader,
    EPOCHS,
    LEARNING_RATE,
    OptimizedQuantumNet,
    TensorDataset,
    mo,
    nn,
    plt,
    torch,
    train_loader,
    x1_test,
    x2_test,
    y_test,
):
    # Initialize optimized model
    model = OptimizedQuantumNet(in_channels=2).to(DEVICE)

    # Use mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # --- Optimized Training Loop ---
    mo.output.append(f"\n--- Training Optimized QuantumNet as Entanglement Witness ---")
    mo.output.append(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    left_norms, right_norms, losses = [], [], []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_batches = 0

        for x1_batch, x2_batch, y_batch in train_loader:
            x1_batch = x1_batch.to(DEVICE, non_blocking=True)
            x2_batch = x2_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output, l_norm, r_norm = model(x1_batch, x2_batch)
                    loss = loss_fn(output, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output, l_norm, r_norm = model(x1_batch, x2_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

        avg_epoch_loss = epoch_loss / epoch_batches
        losses.append(avg_epoch_loss)
        left_norms.append(l_norm.item())
        right_norms.append(r_norm.item())

        scheduler.step(avg_epoch_loss)

        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 100:  # Early stopping
            mo.output.append(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 100 == 0:
            mo.output.append(
                f"Epoch {epoch + 1:4d} | Loss: {avg_epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    # --- Optimized Evaluation ---
    model.eval()
    test_loader = DataLoader(
        TensorDataset(x1_test, x2_test, y_test),
        batch_size=BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        pin_memory=True,
    )

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x1_batch, x2_batch, y_batch in test_loader:
            x1_batch = x1_batch.to(DEVICE, non_blocking=True)
            x2_batch = x2_batch.to(DEVICE, non_blocking=True)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output, _, _ = model(x1_batch, x2_batch)
            else:
                output, _, _ = model(x1_batch, x2_batch)

            predictions = torch.sigmoid(output) > 0.5
            all_predictions.append(predictions.cpu())
            all_targets.append(y_batch)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    accuracy = (all_predictions == all_targets).float().mean().item()

    mo.output.append(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    mo.output.append(f"Total epochs trained: {len(losses)}")

    # --- Visualization ---
    fig = plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(left_norms, label="Path Norm (Full System ρ)", alpha=0.7)
    plt.plot(right_norms, label="Path Norm (Separated ρ_A⊗ρ_B)", alpha=0.7)
    plt.title("Pathway Norms on Entanglement Task")
    plt.xlabel("Epoch")
    plt.ylabel("Pathway Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.output.append(fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🌀 Quantum Entanglement Detection with ChiralNet

    This notebook demonstrates using a **ChiralNet architecture** to detect quantum entanglement in two-qubit systems.
    The key insight is that entangled quantum states cannot be written as a simple product of individual qubit states,
    creating a "chirality" or asymmetry that our network can learn to detect. This is a very simple example of a 
    quantum machine learning model.

    ## 🧮 The Physics Problem

    Given a quantum density matrix ρ for a two-qubit system:
    - **Separable states**: ρ = ρ_A ⊗ ρ_B (can be factored)
    - **Entangled states**: ρ ≠ ρ_A ⊗ ρ_B (cannot be factored)

    Our ChiralNet compares the full density matrix with its "separated" version to detect this fundamental asymmetry.

    ## 🏗️ Architecture Strategy

    The network processes two inputs:
    1. **Full quantum state** ρ (potentially entangled)
    2. **Separated version** ρ_A ⊗ ρ_B (always separable)

    By learning the **difference** between these representations, it becomes an effective **entanglement witness**.
    """
    )
    return


@app.cell
def _(SUBSYSTEM_DIM, SYSTEM_DIM, mo, np, torch):
    # --- Improved Concurrence Calculation for Two-Qubit Systems ---
    def calculate_concurrence_two_qubit(rho_batch):
        """
        Calculate concurrence for a batch of 4x4 two-qubit density matrices.
        For a two-qubit system, concurrence C(ρ) = max(0, λ1 - λ2 - λ3 - λ4)
        where λi are the square roots of eigenvalues of ρ * (σy⊗σy) * ρ* * (σy⊗σy)
        in decreasing order.
        """
        batch_size = rho_batch.shape[0]
        device = rho_batch.device

        # Pauli-Y matrices
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=device)
        # Kronecker product σy ⊗ σy
        sigma_y_kron = torch.kron(sigma_y, sigma_y)

        concurrence_values = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            rho = rho_batch[i]

            # Ensure rho is Hermitian and normalized
            rho = (rho + rho.conj().T) / 2.0
            trace = torch.trace(rho)
            # Take real part of trace for comparison (should be real for valid density matrices)
            trace_real = trace.real
            if not torch.isclose(trace_real, torch.tensor(1.0, device=device)):
                rho = rho / trace

            # Calculate R = ρ * (σy⊗σy) * ρ* * (σy⊗σy)
            rho_star = rho.conj()  # Complex conjugate
            R = rho @ sigma_y_kron @ rho_star @ sigma_y_kron

            # Get eigenvalues of R and take square roots
            eigenvals = torch.linalg.eigvals(R)
            sqrt_eigenvals = torch.sqrt(eigenvals.real.clamp(min=0))  # Ensure non-negative

            # Sort in decreasing order
            sqrt_eigenvals_sorted = torch.sort(sqrt_eigenvals, descending=True)[0]

            # Concurrence formula
            if len(sqrt_eigenvals_sorted) >= 4:
                concurrence = torch.max(
                    torch.tensor(0.0, device=device),
                    sqrt_eigenvals_sorted[0] - sqrt_eigenvals_sorted[1] - 
                    sqrt_eigenvals_sorted[2] - sqrt_eigenvals_sorted[3]
                )
            else:
                concurrence = torch.tensor(0.0, device=device)

            concurrence_values[i] = concurrence.real

        return concurrence_values.unsqueeze(-1)  # Return as column vector

    # --- Helper Functions for Regression Data Generation ---
    def get_random_density_matrices_batch_reg(batch_size, dim, device=None):
        """Vectorized generation of random valid density matrices."""
        device = device or torch.device("cpu")
        mat = torch.randn(batch_size, dim, dim, dtype=torch.cfloat, device=device)
        psd_mat = torch.bmm(mat, mat.conj().transpose(-2, -1))
        traces = torch.diagonal(psd_mat, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        return psd_mat / traces

    def partial_trace_batch_reg(rho_batch, keep_subsystem=0):
        """Vectorized partial trace calculation for batch of 4x4 density matrices."""
        batch_size = rho_batch.shape[0]
        rho_tensor = rho_batch.view(batch_size, 2, 2, 2, 2)
        if keep_subsystem == 0:  # Trace out B to keep A
            reduced_rho = torch.einsum("bijik->bjk", rho_tensor)
        else:  # Trace out A to keep B
            reduced_rho = torch.einsum("bjiki->bjk", rho_tensor)
        return reduced_rho

    # --- Data Generation for Regression ---
    def generate_entanglement_data_regression(n_samples, device=None):
        """Generate quantum states with known concurrence values for regression."""
        device = device or torch.device("cpu")
        print(f"Generating {n_samples} quantum states for concurrence regression...")

        half_samples = n_samples // 2

        # --- Generate Separable States (Concurrence = 0) ---
        print("Generating separable states (concurrence ≈ 0)...")
        rho_A_sep = get_random_density_matrices_batch_reg(half_samples, SUBSYSTEM_DIM, device)
        rho_B_sep = get_random_density_matrices_batch_reg(half_samples, SUBSYSTEM_DIM, device)

        rho_separable = torch.zeros(half_samples, SYSTEM_DIM, SYSTEM_DIM, dtype=torch.cfloat, device=device)
        for i in range(SUBSYSTEM_DIM):
            for j in range(SUBSYSTEM_DIM):
                for k in range(SUBSYSTEM_DIM):
                    for l in range(SUBSYSTEM_DIM):
                        rho_separable[:, i * SUBSYSTEM_DIM + j, k * SUBSYSTEM_DIM + l] = rho_A_sep[:, i, k] * rho_B_sep[:, j, l]

        # --- Generate Entangled States with Varying Concurrence ---
        mo.output.append("Generating entangled states with varying concurrence...")
        # Create maximally entangled Bell states with different mixing parameters
        bell_states = [
            torch.tensor([1, 0, 0, 1], dtype=torch.cfloat, device=device) / np.sqrt(2),  # |Φ+⟩
            torch.tensor([1, 0, 0, -1], dtype=torch.cfloat, device=device) / np.sqrt(2), # |Φ-⟩
            torch.tensor([0, 1, 1, 0], dtype=torch.cfloat, device=device) / np.sqrt(2),  # |Ψ+⟩
            torch.tensor([0, 1, -1, 0], dtype=torch.cfloat, device=device) / np.sqrt(2), # |Ψ-⟩
        ]
        mo.output.append(f"Bell states {bell_states}")

        rho_entangled_batch = torch.zeros(half_samples, SYSTEM_DIM, SYSTEM_DIM, dtype=torch.cfloat, device=device)

        for i in range(half_samples):
            # Choose a random Bell state
            bell_idx = i % len(bell_states)
            bell_state = bell_states[bell_idx]
            rho_bell = torch.outer(bell_state, bell_state.conj())

            # Add random noise to reduce entanglement
            noise_strength = torch.rand(1, device=device).item() * 0.5  # 0 to 50% noise
            noise = get_random_density_matrices_batch_reg(1, SYSTEM_DIM, device).squeeze(0)

            # Mix Bell state with noise
            rho_noisy = (1 - noise_strength) * rho_bell + noise_strength * noise

            # Renormalize
            trace = torch.trace(rho_noisy)
            rho_entangled_batch[i] = rho_noisy / trace

        # Calculate actual concurrence values
        mo.output.append("Calculating concurrence values...")
        concurrence_separable = calculate_concurrence_two_qubit(rho_separable)
        concurrence_entangled = calculate_concurrence_two_qubit(rho_entangled_batch)

        # Create separated versions for ChiralNet comparison
        rho_A_ent = partial_trace_batch_reg(rho_entangled_batch, keep_subsystem=0)
        rho_B_ent = partial_trace_batch_reg(rho_entangled_batch, keep_subsystem=1)

        rho_ent_separated = torch.zeros(half_samples, SYSTEM_DIM, SYSTEM_DIM, dtype=torch.cfloat, device=device)
        for i in range(SUBSYSTEM_DIM):
            for j in range(SUBSYSTEM_DIM):
                for k in range(SUBSYSTEM_DIM):
                    for l in range(SUBSYSTEM_DIM):
                        rho_ent_separated[:, i * SUBSYSTEM_DIM + j, k * SUBSYSTEM_DIM + l] = rho_A_ent[:, i, k] * rho_B_ent[:, j, l]

        # Combine all data
        x1_data = torch.cat([rho_separable, rho_entangled_batch], dim=0)
        x2_data = torch.cat([rho_separable, rho_ent_separated], dim=0)
        y_data = torch.cat([concurrence_separable, concurrence_entangled], dim=0)

        # Convert to real/imag channels
        x1_real = torch.stack([x1_data.real, x1_data.imag], dim=1)
        x2_real = torch.stack([x2_data.real, x2_data.imag], dim=1)

        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        return x1_real[perm], x2_real[perm], y_data[perm]
    return (generate_entanglement_data_regression,)


@app.cell
def _(nn):
    # --- Regression Model ---
    class QuantumConcurrenceNet(nn.Module):
        """Neural network for predicting quantum concurrence."""

        def __init__(self, in_channels=2):
            super().__init__()

            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(64 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
            )

            self.regressor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output between 0 and 1
            )

        def forward(self, x1, x2):
            feat1 = self.feature_extractor(x1)
            feat2 = self.feature_extractor(x2)
            diff_features = feat1 - feat2
            concurrence_pred = self.regressor(diff_features)
            return concurrence_pred, feat1.norm(dim=1).mean(), feat2.norm(dim=1).mean()
    return (QuantumConcurrenceNet,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🔬 Quantum Concurrence Regression with ChiralNet

    Building on the binary classification approach, we now tackle **continuous prediction** of quantum entanglement strength using **concurrence** as our target metric.

    ## 📊 What is Concurrence?

    **Concurrence** is a quantitative measure of entanglement for two-qubit systems:
    - **C = 0**: Completely separable (no entanglement)
    - **C = 1**: Maximally entangled (Bell states)
    - **0 < C < 1**: Partially entangled

    For a two-qubit density matrix ρ, concurrence is calculated as:
    ```
    C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    ```
    where λᵢ are square roots of eigenvalues of ρ(σᵧ⊗σᵧ)ρ*(σᵧ⊗σᵧ) in decreasing order.

    ## 🎯 Regression Approach

    ### Data Generation Strategy:
    1. **Separable States**: Pure product states ρ = ρ_A ⊗ ρ_B → C ≈ 0
    2. **Entangled States**: Noisy Bell states with varying mixing parameters → 0 < C ≤ 1
    3. **Ground Truth**: Calculate actual concurrence for each generated state

    ### Model Architecture:
    - **Input**: Two 4×4 density matrices (real/imaginary channels)
    - **ChiralNet Design**: Compare full state ρ vs separated ρ_A⊗ρ_B  
    - **Output**: Single value ∈ [0,1] via Sigmoid activation
    - **Loss**: Mean Squared Error (MSE)

    ### Key Improvements:
    - ✅ **Proper Concurrence Calculation**: Using the standard quantum formula
    - ✅ **Diverse Bell States**: |Φ±⟩ and |Ψ±⟩ with random noise levels
    - ✅ **Regression Metrics**: MSE, MAE instead of classification accuracy
    - ✅ **Rich Visualization**: Prediction scatter plots, residual analysis, error distributions

    This approach provides **fine-grained entanglement quantification** rather than just binary detection.
    """
    )
    return


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    DataLoader,
    LEARNING_RATE,
    QuantumConcurrenceNet,
    TensorDataset,
    generate_entanglement_data_regression,
    mo,
    nn,
    plt,
    torch,
    train_test_split,
):
    def run_regression_experiment():
        """Run the complete regression experiment in isolated scope."""
        mo.output.append("=== Quantum Concurrence Regression ===")
        torch.manual_seed(42)
        N_SAMPLES_REG = 8192

        # Generate regression data
        x1_reg, x2_reg, y_reg = generate_entanglement_data_regression(N_SAMPLES_REG, DEVICE)

        mo.output.append(f"Generated data shapes: x1={x1_reg.shape}, x2={x2_reg.shape}, y={y_reg.shape}")
        mo.output.append(f"Concurrence range: {y_reg.min().item():.3f} to {y_reg.max().item():.3f}")

        # Move to CPU for train/test split
        if DEVICE.type == "cuda":
            x1_cpu_reg, x2_cpu_reg, y_cpu_reg = x1_reg.cpu(), x2_reg.cpu(), y_reg.cpu()
        else:
            x1_cpu_reg, x2_cpu_reg, y_cpu_reg = x1_reg, x2_reg, y_reg

        # Train/test split
        indices_reg = list(range(N_SAMPLES_REG))
        train_indices_reg, test_indices_reg = train_test_split(
            indices_reg, test_size=0.2, random_state=42
        )

        x1_train_reg = x1_cpu_reg[train_indices_reg]
        x2_train_reg = x2_cpu_reg[train_indices_reg]
        y_train_reg = y_cpu_reg[train_indices_reg]

        x1_test_reg = x1_cpu_reg[test_indices_reg]
        x2_test_reg = x2_cpu_reg[test_indices_reg]
        y_test_reg = y_cpu_reg[test_indices_reg]

        # Create data loaders
        train_loader_reg = DataLoader(
            TensorDataset(x1_train_reg, x2_train_reg, y_train_reg),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=2 if torch.cuda.is_available() else 0,
        )

        test_loader_reg = DataLoader(
            TensorDataset(x1_test_reg, x2_test_reg, y_test_reg),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=True,
        )

        # Initialize model
        model_reg = QuantumConcurrenceNet(in_channels=2).to(DEVICE)
        optimizer_reg = torch.optim.AdamW(model_reg.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler_reg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_reg, patience=50, factor=0.5)
        loss_fn_reg = nn.MSELoss()

        print(f"Model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")

        # Training loop
        print("\n--- Training Concurrence Regression Model ---")
        losses_reg = []
        left_norms_reg = []
        right_norms_reg = []
        best_loss_reg = float("inf")
        patience_counter_reg = 0

        for epoch_reg in range(500):
            model_reg.train()
            epoch_loss_reg = 0
            epoch_batches_reg = 0

            for x1_batch, x2_batch, y_batch in train_loader_reg:
                x1_batch = x1_batch.to(DEVICE, non_blocking=True)
                x2_batch = x2_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)

                optimizer_reg.zero_grad()
                concurrence_pred, l_norm_reg, r_norm_reg = model_reg(x1_batch, x2_batch)
                loss_reg = loss_fn_reg(concurrence_pred, y_batch)

                loss_reg.backward()
                optimizer_reg.step()

                epoch_loss_reg += loss_reg.item()
                epoch_batches_reg += 1

            avg_epoch_loss_reg = epoch_loss_reg / epoch_batches_reg
            losses_reg.append(avg_epoch_loss_reg)
            left_norms_reg.append(l_norm_reg.item())
            right_norms_reg.append(r_norm_reg.item())

            scheduler_reg.step(avg_epoch_loss_reg)

            # Early stopping
            if avg_epoch_loss_reg < best_loss_reg:
                best_loss_reg = avg_epoch_loss_reg
                patience_counter_reg = 0
            else:
                patience_counter_reg += 1

            if patience_counter_reg >= 100:
                mo.output.append(f"Early stopping at epoch {epoch_reg + 1}")
                break

            if (epoch_reg + 1) % 100 == 0:
                mo.output.append(f"Epoch {epoch_reg + 1:4d} | Loss: {avg_epoch_loss_reg:.6f} | LR: {optimizer_reg.param_groups[0]['lr']:.2e}")

        # Evaluation
        model_reg.eval()
        all_predictions_reg = []
        all_targets_reg = []

        with torch.no_grad():
            for x1_batch, x2_batch, y_batch in test_loader_reg:
                x1_batch = x1_batch.to(DEVICE, non_blocking=True)
                x2_batch = x2_batch.to(DEVICE, non_blocking=True)

                output_reg, _, _ = model_reg(x1_batch, x2_batch)
                all_predictions_reg.append(output_reg.cpu())
                all_targets_reg.append(y_batch)

        all_predictions_reg = torch.cat(all_predictions_reg)
        all_targets_reg = torch.cat(all_targets_reg)

        # Calculate regression metrics
        mse = nn.MSELoss()(all_predictions_reg, all_targets_reg).item()
        mae = nn.L1Loss()(all_predictions_reg, all_targets_reg).item()

        mo.output.append(f"\n=== Regression Results ===")
        mo.output.append(f"Test MSE: {mse:.6f}")
        mo.output.append(f"Test MAE: {mae:.6f}")
        mo.output.append(f"Prediction range: {all_predictions_reg.min().item():.3f} to {all_predictions_reg.max().item():.3f}")
        mo.output.append(f"Target range: {all_targets_reg.min().item():.3f} to {all_targets_reg.max().item():.3f}")

        # --- Comprehensive Visualization ---
        fig = plt.figure(figsize=(18, 12))

        # Training loss
        plt.subplot(2, 3, 1)
        plt.plot(losses_reg)
        plt.title("Training Loss (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        # Feature norms
        plt.subplot(2, 3, 2)
        plt.plot(left_norms_reg, label="Full System ρ", alpha=0.7)
        plt.plot(right_norms_reg, label="Separated ρ_A⊗ρ_B", alpha=0.7)
        plt.title("Feature Pathway Norms")
        plt.xlabel("Epoch")
        plt.ylabel("Norm")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Predictions vs targets scatter plot
        plt.subplot(2, 3, 3)
        plt.scatter(all_targets_reg.numpy(), all_predictions_reg.numpy(), alpha=0.6, s=10)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        plt.xlabel("True Concurrence")
        plt.ylabel("Predicted Concurrence")
        plt.title("Predictions vs Truth")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Residuals plot
        plt.subplot(2, 3, 4)
        residuals = all_predictions_reg.numpy() - all_targets_reg.numpy()
        plt.scatter(all_targets_reg.numpy(), residuals, alpha=0.6, s=10)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("True Concurrence")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Truth")
        plt.grid(True, alpha=0.3)

        # Distribution of predictions and targets
        plt.subplot(2, 3, 5)
        plt.hist(all_targets_reg.numpy(), bins=50, alpha=0.7, label="True", density=True)
        plt.hist(all_predictions_reg.numpy(), bins=50, alpha=0.7, label="Predicted", density=True)
        plt.xlabel("Concurrence")
        plt.ylabel("Density")
        plt.title("Distribution Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error distribution
        plt.subplot(2, 3, 6)
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title("Error Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        mo.output.append(fig)

        return model_reg, losses_reg, left_norms_reg, right_norms_reg, all_predictions_reg, all_targets_reg, mse, mae

    # Run the experiment
    result_regression = run_regression_experiment()

    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
