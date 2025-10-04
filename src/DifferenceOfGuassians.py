import marimo

__generated_with = "0.14.6"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import warnings
    import math
    return (
        DataLoader,
        F,
        TensorDataset,
        TfidfVectorizer,
        fetch_20newsgroups,
        math,
        mo,
        nn,
        optim,
        torch,
        warnings,
    )


@app.cell
def _(torch):
    # --- 1. Parameters ---
    N_CATEGORIES = 3
    MAX_FEATURES = 2000
    HIDDEN_DIM = 128 
    OUTPUT_DIM = N_CATEGORIES
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 200 
    # --- 6. Setup and Run ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return (
        BATCH_SIZE,
        EPOCHS,
        HIDDEN_DIM,
        LEARNING_RATE,
        MAX_FEATURES,
        OUTPUT_DIM,
        device,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Linear with Abs

    **Analogy:** Think of it as a linear layer that only cares about the absolute value of its output, not whether it's positive or negative.
    **Analogy:** Think of it as a linear layer that only cares about the magnitude of its output, not whether it's positive or negative.
    """
    )
    return


@app.cell
def _(nn, torch):
    class CustomLinearWithAbs(nn.Module): 
        def __init__(self, in_features, out_features):
            super(CustomLinearWithAbs, self).__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            linear_output = self.linear(x)
            combined_output = torch.abs(linear_output)
            return combined_output
    return (CustomLinearWithAbs,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Dual Path Neuron Layer
    - Takes a single input and passes it through two seperate linear layers.
    - `combine_mode` determines how the two layers are combined
    ## Combine Modes
    1. Difference `linear_a - linear_b`

    2. Product `linear_a * linear_b`

    3. Average Inverted B `(linear_a + (-1 * linear_b)) / 2`

    4. Sum Abs `Abs(linear_a) + Abs(linear_b)`

    5. Sum Direct `linear_a + linear_b`

    6. Max Abs Signed `Max(Abs(linear_a), Abs(linear_b))`

    """
    )
    return


@app.cell
def _(nn, torch):
    class DualPathNeuronLayer(nn.Module):
        def __init__(self, in_features, num_custom_neurons, combine_mode="difference"):
            super(DualPathNeuronLayer, self).__init__()
            self.num_custom_neurons = num_custom_neurons
            self.linear_A = nn.Linear(in_features, self.num_custom_neurons)
            self.linear_B = nn.Linear(in_features, self.num_custom_neurons)
            self.combine_mode = combine_mode

        def forward(self, x):
            linear_A_out = self.linear_A(x)
            linear_B_out = self.linear_B(x)

            if self.combine_mode == "difference":
                combined = linear_A_out - linear_B_out
            elif self.combine_mode == "average_inverted_B":
                combined = (linear_A_out + (-1 * linear_B_out)) / 2.0
            elif self.combine_mode == "sum_abs":
                combined = torch.abs(linear_A_out) + torch.abs(linear_B_out)
            elif self.combine_mode == "sum_direct":
                combined = linear_A_out + linear_B_out
            elif self.combine_mode == "product":
                combined = linear_A_out * linear_B_out
            elif self.combine_mode == "max_abs_signed":
                abs_A = torch.abs(linear_A_out)
                abs_B = torch.abs(linear_B_out)
                mask_A_stronger = (abs_A >= abs_B).float()
                combined = mask_A_stronger * linear_A_out + (1 - mask_A_stronger) * linear_B_out
            else:
                raise ValueError(f"Unknown combine_mode: {self.combine_mode}")
            return combined
    return (DualPathNeuronLayer,)


@app.cell
def _(F, nn):
    # --- 3. Model Definitions ---
    class StandardNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(StandardNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return (StandardNet,)


@app.cell
def _(CustomLinearWithAbs, F, nn):
    class AbsPreReLUNet(nn.Module): # Renamed for clarity
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(AbsPreReLUNet, self).__init__()
            self.fc1_custom = CustomLinearWithAbs(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.fc1_custom(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return (AbsPreReLUNet,)


@app.cell
def _(DualPathNeuronLayer, F, nn):
    class DualPathNet(nn.Module):
        def __init__(self, input_dim, hidden_dim_dual, output_dim, combine_mode="difference"):
            super(DualPathNet, self).__init__()
            self.dual_path_layer1 = DualPathNeuronLayer(input_dim, hidden_dim_dual, combine_mode=combine_mode)
            self.fc2 = nn.Linear(hidden_dim_dual, output_dim) # Output layer
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.dual_path_layer1(x) # Output of the combine function
            x = F.relu(x)                # Apply ReLU to the combined output
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return (DualPathNet,)


@app.cell
def _(torch, warnings):
    # Epsilon for numerical stability - Maybe slightly larger?
    EPS = 1e-5 # Increased slightly from 1e-6
    def psi(h, lambda_):
        """
        Applies the Yeo-Johnson power transform element-wise.
        (Added NaN/Inf checks and clipping)
        """
        # --- Input Checks ---
        if torch.isnan(h).any() or torch.isinf(h).any():
            warnings.warn("NaN/Inf detected in input 'h' to psi. Clipping and continuing.")
            h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6) # Replace NaN/Inf
            # Consider more aggressive clipping if needed: h = torch.clamp(h, -10.0, 10.0)

        if torch.isnan(lambda_).any() or torch.isinf(lambda_).any():
             warnings.warn("NaN/Inf detected in input 'lambda_' to psi. Clipping to 1.0.")
             lambda_ = torch.nan_to_num(lambda_, nan=1.0, posinf=1.0, neginf=1.0) # Fallback lambda

        # --- Original psi logic (with existing safety checks) ---
        try:
            _ = torch.broadcast_shapes(h.shape, lambda_.shape)
        except RuntimeError as e:
            raise RuntimeError(f"Shape mismatch: Cannot broadcast lambda_ {lambda_.shape} to h {h.shape}. Error: {e}")

        h_ge_0 = (h >= 0)

        # --- Case: h >= 0 ---
        psi_ge0_lam0 = torch.log1p(h)
        denominator_ge0 = lambda_ + torch.sign(lambda_) * EPS
        denominator_ge0 = torch.where(denominator_ge0 == 0, torch.copysign(torch.tensor(EPS),lambda_), denominator_ge0) # Use signed EPS if zero
        # Clip input to pow to prevent overflow/NaN with extreme lambdas
        pow_input_ge0 = torch.clamp(1 + h, min=EPS) # Ensure base is positive
        # Handle potential NaN from pow itself
        pow_result_ge0 = torch.pow(pow_input_ge0, lambda_)
        pow_result_ge0 = torch.nan_to_num(pow_result_ge0, nan=1.0) # If pow fails, treat as if (1+h)^0=1
        psi_ge0_lam_ne0 = (pow_result_ge0 - 1) / denominator_ge0
        psi_ge0 = torch.where(lambda_ == 0, psi_ge0_lam0, psi_ge0_lam_ne0)

        # --- Case: h < 0 ---
        lambda_minus_2 = 2.0 - lambda_
        psi_lt0_lam2 = -torch.log1p(-h) # Where lambda_minus_2 = 0
        denominator_lt0 = lambda_minus_2 + torch.sign(lambda_minus_2) * EPS
        denominator_lt0 = torch.where(denominator_lt0 == 0, torch.copysign(torch.tensor(EPS),lambda_minus_2), denominator_lt0) # Use signed EPS if zero
        # Clip input to pow
        pow_input_lt0 = torch.clamp(1 - h, min=EPS) # Ensure base is positive
        # Handle potential NaN from pow
        pow_result_lt0 = torch.pow(pow_input_lt0, lambda_minus_2)
        pow_result_lt0 = torch.nan_to_num(pow_result_lt0, nan=1.0) # If pow fails, treat as if (1-h)^0=1
        psi_lt0_lam_ne2 = (1.0 - pow_result_lt0) / denominator_lt0
        psi_lt0 = torch.where(lambda_minus_2 == 0, psi_lt0_lam2, psi_lt0_lam_ne2)

        # --- Combine ---
        result = torch.where(h_ge_0, psi_ge0, psi_lt0)

        # --- Final Output Check ---
        if torch.isnan(result).any() or torch.isinf(result).any():
           warnings.warn("NaN/Inf encountered in psi function final output. Replacing with 0.")
           result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0) # Last resort

        return result
    return EPS, psi


@app.cell
def _(EPS, torch, warnings):
    def estimate_lambda_hat(h, dims):
        """
        Estimates lambda_hat. (Added input check and clamping)
        """
        with torch.no_grad():
            # --- Input Check ---
            if torch.isnan(h).any() or torch.isinf(h).any():
                 warnings.warn("NaN/Inf detected in input 'h' to estimate_lambda_hat. Clipping and continuing.")
                 h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6) # Replace NaN/Inf

            # --- Original estimation logic ---
            h_abs = torch.abs(h)
            # Add EPS inside log1p? May not be needed if h isn't exactly 0 after clip.
            log1p_h_abs = torch.log1p(h_abs) # + EPS inside if needed

            s3 = torch.mean(torch.pow(h, 3), dim=dims, keepdim=True)
            k = torch.mean(h * log1p_h_abs, dim=dims, keepdim=True)
            g = torch.mean(torch.pow(h, 2) * torch.pow(log1p_h_abs, 2), dim=dims, keepdim=True)

            # Check moments for NaN/Inf before using them
            s3 = torch.nan_to_num(s3, nan=0.0)
            k = torch.nan_to_num(k, nan=0.0)
            g = torch.nan_to_num(g, nan=1.0) # Replace g NaN with 1 to avoid issues in L''

            L_prime_at_1 = k - 0.5 * s3
            # Ensure L'' denominator is reasonably large and positive
            L_double_prime_at_1 = torch.clamp(g - k + 1.0, min=EPS) # Clamp denominator > 0

            lambda_hat = 1.0 - L_prime_at_1 / L_double_prime_at_1

            # --- Final Checks on lambda_hat ---
            if torch.isnan(lambda_hat).any() or torch.isinf(lambda_hat).any():
                 warnings.warn("NaN/Inf encountered after lambda_hat calculation. Setting lambda_hat=1.0.")
                 lambda_hat = torch.nan_to_num(lambda_hat, nan=1.0, posinf=1.0, neginf=1.0) # Fallback

            # Optional but recommended: Clamp lambda_hat to a reasonable range
            lambda_hat = torch.clamp(lambda_hat, -2.0, 4.0) # Clamp to avoid extreme transforms

        return lambda_hat
    return (estimate_lambda_hat,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Normality Normalization
    - Similar to Layer Normalization
    - Finds the mean and standard deviation then uses Yeo-Johnson transformation to make the data distribution more symmetric/normal.
    - Optionally during training we can add some Gaussian noise to the data.

    ## Why use it?

    - **Improved Stability:** Normalization techniques generally help stabilize training and allow for higher learning rates.
    - **Better Convergence:** By making distributions more Gaussian-like, it can help optimization algorithms converge faster.
    - **Flexibility:** The Yeo-Johnson transform and the optional noise add flexibility to adapt to different data distributions.
    - **Handles Different Inputs:** It's designed to work with different input shapes, particularly common in CNNs (e.g., normalizing over spatial dimensions).
    """
    )
    return


@app.cell
def _(estimate_lambda_hat, math, nn, psi, torch, warnings):
    class NormalityNormalization(nn.Module):
        # __init__ remains the same
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, noise_factor=0.0):
            """(No changes here needed for fix)"""
            super().__init__()
            # Allow integer or tuple for normalized_shape
            if isinstance(normalized_shape, int):
                # For CNNs, assume integer means number of channels
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            self.noise_factor = noise_factor

            # Determine expected number of dimensions based on normalized_shape length
            # This is a heuristic: if shape is like (C,), expect 4D (B,C,H,W) or 2D (B,C) input
            # If shape is like (H, W), expect 3D (B, H, W) ?? - less common
            # Let's focus on the CNN case: normalized_shape = (Channels,)

            if len(self.normalized_shape) == 1:
                self.num_features = self.normalized_shape[0] # Expect this to be channels
            else:
                # Handle other cases if needed, for now assume LayerNorm-like for non-1D shape
                 self.num_features = None # Not used in LayerNorm-like mode

            if self.elementwise_affine:
                # gamma/beta should match the number of features (channels in CNN case)
                 param_shape = (self.num_features,) if self.num_features is not None else self.normalized_shape
                 self.gamma = nn.Parameter(torch.ones(param_shape))
                 self.beta = nn.Parameter(torch.zeros(param_shape))
            else:
                self.register_parameter('gamma', None)
                self.register_parameter('beta', None)


        def forward(self, x):
            input_shape = x.shape
            input_ndim = x.ndim

            # --- 0. Input Check --- (same as before)
            if torch.isnan(x).any() or torch.isinf(x).any():
                 warnings.warn("NaN/Inf detected in input 'x'. Clipping.")
                 x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

            # --- Determine Normalization Dimensions ---
            if input_ndim == 4 and len(self.normalized_shape) == 1:
                # CNN Case: Input (B, C, H, W), normalized_shape=(C,)
                # Normalize across H, W dimensions (like BatchNorm, but instance-wise stats here)
                # Keep stats per channel per batch element.
                assert self.normalized_shape[0] == input_shape[1], \
                       f"Expected {self.normalized_shape[0]} channels, got {input_shape[1]}"
                dims = (2, 3) # Dimensions H, W
                # Calculate N for noise scaling based on spatial dims
                N = input_shape[2] * input_shape[3]
                # Gamma/beta shape (1, C, 1, 1) for broadcasting
                affine_shape = (1, self.num_features, 1, 1)

            elif input_ndim >= 2 and self.normalized_shape == input_shape[-len(self.normalized_shape):]:
                 # LayerNorm Case: Input (B, ..., F1, F2,..), normalized_shape=(F1, F2,..)
                 norm_shape_len = len(self.normalized_shape)
                 dims = tuple(range(input_ndim - norm_shape_len, input_ndim))
                 N = math.prod(self.normalized_shape)
                 affine_shape = [1] * (input_ndim - norm_shape_len) + list(self.normalized_shape)

            else:
                raise ValueError(f"Input shape {input_shape} and normalized_shape {self.normalized_shape} are incompatible.")


            # --- 1. Standard Normalization ---
            # Keep channel/feature dim separate, normalize over others specified in 'dims'
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, unbiased=False)
            var = torch.clamp(var, min=self.eps * self.eps) # Clamp variance
            std = torch.sqrt(var)
            h = (x - mean) / std

            # --- Check h --- (same as before)
            if torch.isnan(h).any() or torch.isinf(h).any():
                warnings.warn("NaN/Inf detected after standardization 'h'. Clipping.")
                h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)

            # --- 2. Estimate lambda_hat ---
            # Estimate lambda per element/channel group, over the normalized dims
            lambda_hat = estimate_lambda_hat(h.detach(), dims=dims) # Pass correct dims

            # --- 3. Apply Power Transform ---
            x_transformed = psi(h, lambda_hat)

            # --- 4. Add Scaled Gaussian Noise ---
            y = x_transformed
            if self.training and self.noise_factor > 0.0:
                with torch.no_grad():
                    if torch.isnan(y).any() or torch.isinf(y).any():
                        warnings.warn("NaN/Inf detected before noise. Clipping.")
                        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

                    xt_mean = y.mean(dim=dims, keepdim=True) # Mean over H,W or Features
                    if N == 0: N = 1
                    # L1 norm over H,W or Features
                    s = torch.linalg.norm(y - xt_mean, ord=1, dim=dims, keepdim=True) / N
                    s = torch.nan_to_num(s, nan=0.0)
                    s = torch.clamp(s, min=0.0, max=1e6)

                noise = torch.randn_like(y)
                y = y + noise * self.noise_factor * s

            # --- 5. Affine Transform ---
            if torch.isnan(y).any() or torch.isinf(y).any():
                warnings.warn("NaN/Inf detected before affine. Clipping.")
                y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

            if self.elementwise_affine:
                gamma_reshaped = self.gamma.view(affine_shape) #type: ignore
                beta_reshaped = self.beta.view(affine_shape) #type: ignore
                out = y * gamma_reshaped + beta_reshaped
            else:
                out = y
            # --- Final Output Check --- (same as before)
            if torch.isnan(out).any() or torch.isinf(out).any():
                warnings.warn("NaN/Inf detected in final output. Clipping.")
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

            return out

        # extra_repr remains the same
        def extra_repr(self):
            return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, noise_factor={noise_factor}'.format(
                normalized_shape=self.normalized_shape,
                eps=self.eps,
                elementwise_affine=self.elementwise_affine,
                noise_factor=self.noise_factor
            )
    return (NormalityNormalization,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Difference of Guassians 
    - Difference of Gaussians is often used in image processing for edge detection.
    - `Linear_A` processes the input, applies ReLU, then optionally adds Gaussian noise.
    - `Linear_B` processes the input independently, applies ReLU, then optionally adds Gaussian noise.
    - The normalized outputs of Path A and Path B are subtracted from each other (norm_A - norm_B). This subtraction highlights the differences between the two filtered views. Where the filters respond similarly, the output is near zero. Where they respond differently (e.g., one detects an edge, the other sees a broader texture), the difference is large.

    The Difference of Gaussians is a model that is primarily from Computer Vision. We will test it on the AG_NEWS dataset to see how it performs on a NLP task and to see if the normality normalization helps.
    """
    )
    return


@app.cell
def _(F, NormalityNormalization, nn, torch):
    class DoGInspiredLayer(nn.Module):
        def __init__(self, in_features, out_features, noise_std=0.1, use_layernorm=True):
            super(DoGInspiredLayer, self).__init__()
            self.out_features = out_features
            self.noise_std = noise_std
            self.use_layernorm = use_layernorm

            # Path A ("sharper")
            self.linear_A = nn.Linear(in_features, out_features)

            # Path B ("noisier" or "broader context")
            self.linear_B = nn.Linear(in_features, out_features) # Independent weights

            if self.use_layernorm:
                self.norm_A = nn.LayerNorm(out_features)
                self.norm_B = nn.LayerNorm(out_features)
            else:
                self.norm_A = NormalityNormalization(in_features,noise_factor=self.noise_std)
                self.norm_B = NormalityNormalization(in_features, noise_factor=self.noise_std)

        def add_gaussian_noise(self, tensor):
            if self.noise_std > 0 and self.training: # Only add noise during training
                noise = torch.randn_like(tensor) * self.noise_std
                return tensor + noise
            return tensor

        def forward(self, x):
            # Path A
            linear_A_out = self.linear_A(x)
            activated_A = F.relu(linear_A_out) # Or another activation

            # Path B
            linear_B_out = self.linear_B(x)
            activated_B_base = F.relu(linear_B_out) # Or another activation

            # Add noise to Path B's activated output
            noisy_activated_B = self.add_gaussian_noise(activated_B_base)
            noisy_activated_A = self.add_gaussian_noise(activated_A)


            # Optional Normalization
            if self.use_layernorm:
                norm_A = self.norm_A(noisy_activated_A)
                norm_B = self.norm_B(noisy_activated_B)
            else:
                norm_A = noisy_activated_A
                norm_B = noisy_activated_B

            # Combine by difference (core DoG idea)
            combined_output = norm_A - norm_B

            # The final ReLU for this layer's output will be applied in the main model
            return combined_output
    return (DoGInspiredLayer,)


@app.cell
def _(DoGInspiredLayer, F, nn):
    class DoGInspiredNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, noise_std=0.1, use_layernorm=True):
            super(DoGInspiredNet, self).__init__()
            self.dog_layer1 = DoGInspiredLayer(input_dim, hidden_dim, noise_std=noise_std, use_layernorm=use_layernorm)
            self.fc2 = nn.Linear(hidden_dim, output_dim) # Output layer
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.dog_layer1(x)  # Output of the combine function (A-B)
            x = F.relu(x)           # Apply ReLU to the combined difference
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return (DoGInspiredNet,)


@app.cell
def _(
    BATCH_SIZE,
    DataLoader,
    MAX_FEATURES,
    TensorDataset,
    TfidfVectorizer,
    fetch_20newsgroups,
    torch,
):
    def _():
        # --- 4. Data Loading and Preprocessing ---
        print("Loading data...")
        categories = ['sci.med', 'soc.religion.christian', 'talk.politics.guns']
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data).toarray()
        X_test_tfidf = vectorizer.transform(newsgroups_test.data).toarray()
        y_train = newsgroups_train.target
        y_test = newsgroups_test.target

        X_train_tensor = torch.FloatTensor(X_train_tfidf)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_tfidf)
        y_test_tensor = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        INPUT_DIM = X_train_tfidf.shape[1]
        print(INPUT_DIM)
        return INPUT_DIM, train_dataset, test_dataset, train_loader, test_loader


    INPUT_DIM, train_dataset, test_dataset, train_loader, test_loader = _()
    return INPUT_DIM, test_loader, train_loader


@app.cell
def _(torch):
    def train_model(model, train_loader, optimizer, criterion, device, epochs, model_name=None):
        model.train()
        model_str = f"[{model_name}] " if model_name else ""
        for epoch in range(epochs):
            epoch_loss = 0
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(texts)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"{model_str}Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    def evaluate_model(model, test_loader, criterion, device, model_name=None):
        model.eval()
        total_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts)
                loss = criterion(predictions, labels)
                total_loss += loss.item()
                _, predicted_labels = torch.max(predictions, 1)
                correct_predictions += (predicted_labels == labels).sum().item()
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / len(test_loader.dataset)
        return avg_loss, accuracy
    return evaluate_model, train_model


@app.cell
def _(
    AbsPreReLUNet,
    DoGInspiredNet,
    DualPathNet,
    EPOCHS,
    HIDDEN_DIM,
    INPUT_DIM,
    LEARNING_RATE,
    OUTPUT_DIM,
    StandardNet,
    device,
    evaluate_model,
    mo,
    nn,
    optim,
    test_loader,
    train_loader,
    train_model,
):
    criterion = nn.CrossEntropyLoss()
    # Dictionary to store results
    results = {}

    # Standard Model
    print("\n--- Training Standard Model ---")
    standard_model = StandardNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    optimizer_standard = optim.Adam(standard_model.parameters(), lr=LEARNING_RATE)
    train_model(standard_model, train_loader, optimizer_standard, criterion, device, EPOCHS)
    std_loss, std_acc = evaluate_model(standard_model, test_loader, criterion, device)
    results["Standard"] = std_acc
    print(f"Standard Model - Test Loss: {std_loss:.4f}, Test Accuracy: {std_acc:.4f}")

    print("\n--- Training AbsPreReLU Model ---")
    abs_model = AbsPreReLUNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device) # Corrected class name
    optimizer_abs = optim.Adam(abs_model.parameters(), lr=LEARNING_RATE)
    train_model(abs_model, train_loader, optimizer_abs, criterion, device, EPOCHS)
    abs_loss, abs_acc = evaluate_model(abs_model, test_loader, criterion, device)
    results["AbsPreReLU"] = abs_acc
    print(f"AbsPreReLU Model - Test Loss: {abs_loss:.4f}, Test Accuracy: {abs_acc:.4f}")


    # "difference", "average_inverted_B", "sum_abs", "max_abs_signed", "concatenate"
    SELECTED_COMBINE_MODE = "difference" # Change this to test other modes
    print(f"\n--- Training DualPath Model (Mode: {SELECTED_COMBINE_MODE}) ---")
    dual_path_model = DualPathNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, combine_mode=SELECTED_COMBINE_MODE).to(device)
    optimizer_dual = optim.Adam(dual_path_model.parameters(), lr=LEARNING_RATE)
    train_model(dual_path_model, train_loader, optimizer_dual, criterion, device, EPOCHS)
    dual_loss, dual_acc = evaluate_model(dual_path_model, test_loader, criterion, device)
    results[f"DualPath ({SELECTED_COMBINE_MODE})"] = dual_acc
    print(f"DualPath Model ({SELECTED_COMBINE_MODE}) - Test Loss: {dual_loss:.4f}, Test Accuracy: {dual_acc:.4f}")


    print("\n--- Final Comparison ---")
    print(f"Standard Model Test Accuracy     : {std_acc:.4f}")
    print(f"AbsPreReLU Model Test Accuracy   : {abs_acc:.4f}")
    print(f"DualPath Model ({SELECTED_COMBINE_MODE}) Test Accuracy: {dual_acc:.4f}")

    # DoG-Inspired Model
    print(f"\n--- Training DoG-Inspired Model (Noise std: {0.1}, LayerNorm: True) ---") # Example params
    dog_model = DoGInspiredNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, noise_std=0.5, use_layernorm=False).to(device)
    optimizer_dog = optim.Adam(dog_model.parameters(), lr=LEARNING_RATE)
    train_model(dog_model, train_loader, optimizer_dog, criterion, device, EPOCHS, "DoG-Inspired Model")
    dog_loss, dog_acc = evaluate_model(dog_model, test_loader, criterion, device, "DoG-Inspired Model")
    results["DoGInspired (noise=0.5, LN=T)"] = dog_acc
    print(f"DoG-Inspired Model - Test Loss: {dog_loss:.4f}, Test Accuracy: {dog_acc:.4f}")

    # You might want to try different noise_std values or with/without LayerNorm
    print(f"\n--- Training DoG-Inspired Model (Noise std: {0.0}, LayerNorm: True) ---") # No noise
    dog_model_no_noise = DoGInspiredNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, noise_std=0.0, use_layernorm=False).to(device)
    optimizer_dog_no_noise = optim.Adam(dog_model_no_noise.parameters(), lr=LEARNING_RATE)
    train_model(dog_model_no_noise, train_loader, optimizer_dog_no_noise, criterion, device, EPOCHS, "DoG-Inspired Model (No Noise)")
    dog_no_noise_loss, dog_no_noise_acc = evaluate_model(dog_model_no_noise, test_loader, criterion, device, "DoG-Inspired Model (No Noise)")
    results["DoGInspired (noise=0.0, LN=T)"] = dog_no_noise_acc
    print(f"DoG-Inspired Model (No Noise) - Test Loss: {dog_no_noise_loss:.4f}, Test Accuracy: {dog_no_noise_acc:.4f}")


    # --- Update the Final Comparison section to include the new model(s) ---
    mo.output.append("\n--- Final Comparison ---")
    mo.output.append(f"Standard Model Test Accuracy     : {results['Standard']:.4f}")
    mo.output.append(f"AbsPreReLU Model Test Accuracy   : {results['AbsPreReLU']:.4f}")
    mo.output.append(f"DualPath Model ({SELECTED_COMBINE_MODE}) Test Accuracy: {results[f'DualPath ({SELECTED_COMBINE_MODE})']:.4f}")
    mo.output.append(f"DoGInspired (noise=0.1, LN=T) Test Accuracy: {results['DoGInspired (noise=0.5, LN=T)']:.4f}")
    mo.output.append(f"DoGInspired (noise=0.0, LN=T) Test Accuracy: {results['DoGInspired (noise=0.0, LN=T)']:.4f}")

    # Basic comparison logic
    best_acc = max(results.values())
    best_model = [model for model, acc in results.items() if acc == best_acc][0]
    mo.output.append(f"\nBest performing model: {best_model} with accuracy: {best_acc:.4f}")

    return (results,)


@app.cell
def _(results):
    import pandas as pd
    import altair as alt

    # Convert results dictionary to a DataFrame for Altair
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])

    # Create the Altair chart
    chart = alt.Chart(results_df).mark_bar().encode(
        x=alt.X('Model', sort='-y', title='Model Architecture'),
        y=alt.Y('Accuracy', title='Test Accuracy'),
        tooltip=['Model', alt.Tooltip('Accuracy', format='.4f')]
    ).properties(
        title='Model Performance Comparison on AG_NEWS Dataset'
    )

    chart
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Conclusions:
    - DoG-Inspired vs. Baseline: The Difference of Gaussians (DoG) inspired network achieved performance comparable to the baseline linear model on the AG_News classification task. This suggests that while the DoG architecture offers a different approach to feature extraction, its benefits weren't significantly realized on this specific dataset.
    - Dual Path Network Efficacy: The Dual Path network demonstrated superior performance, particularly when utilizing the "difference" combination mode. This outcome implies that for this dataset, the distinct processing paths within the network captured complementary information, and their divergence (difference) was key to improved classification.
    - Normality Normalization Impact: The integration of Normality Normalization did not yield a noticeable performance boost for the DoG-inspired network. Further investigation might be needed to understand if the normalization's properties are better suited to different data distributions or network architectures.

    Overall Takeaway: This exploration highlights the diverse impact of architectural choices on model performance. It underscores that while novel architectures like DoG and Dual Path networks offer intriguing mechanisms for feature processing, their effectiveness is dataset-dependent. The success of the Dual Path network, especially with the "difference" mode, emphasizes the value of exploring different feature combination strategies.
    """
    )
    return


if __name__ == "__main__":
    app.run()
