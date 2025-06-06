# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    DataCollatorWithPadding,
)
import numpy as np
import math
import random
import tqdm
from torch.distributions import MultivariateNormal
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform

# %%
def create_normalizing_flow_model(
    num_layers, hidden_features, input_dim
):
    """
    Creates a simple Normalizing Flow model using nflows library.
    """
    transforms = []
    for _ in range(num_layers):
        # Masked Autoregressive Flow (MAF) block
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=input_dim,
                hidden_features=hidden_features,
                num_blocks=2,  # Number of residual blocks in the MAF's internal NN
                use_residual_blocks=True,
                random_mask=False,  # Use alternating masks for better expressiveness
                activation=nn.ReLU(),
            )
        )
    transform = CompositeTransform(transforms)
    base_distribution = StandardNormal(shape=[input_dim])
    flow = Flow(transform, base_distribution)
    return flow


# --- 2. Define the objective function for the Swarm Algorithm ---
# This function will train and evaluate a Normalizing Flow
# based on the hyperparameters suggested by the swarm.
def evaluate_flow_performance(hyperparameters, data):
    """
    Trains a Normalizing Flow with given hyperparameters and returns its NLL.
    This is the objective function for the swarm.

    Args:
        hyperparameters (list/array): [num_layers, hidden_features]
        data (torch.Tensor): Training data for the Normalizing Flow.

    Returns:
        float: Negative Log-Likelihood (NLL) of the trained model on the data.
                Lower is better.
    """
    num_layers = int(hyperparameters[0])
    hidden_features = int(hyperparameters[1])

    # Ensure valid hyperparameters (e.g., minimum 1 layer, some features)
    if num_layers < 1 or hidden_features < 10:
        return float("inf")  # Penalize invalid configurations

    input_dim = data.shape[1]
    model = create_normalizing_flow_model(
        num_layers, hidden_features, input_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simple training loop (for demonstration, a real loop would be longer)
    num_epochs = 50  # Keep it short for a conceptual example
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # NLL is directly available from nflows
        nll = -model.log_prob(data).mean()
        nll.backward()
        optimizer.step()

    # Return the final NLL as the fitness score
    # Swarm algorithms typically minimize, so we return NLL directly.
    return nll.item()

# %% 
class BasicPSO:
    def __init__(
        self,
        objective_function,
        bounds,
        num_particles=10,
        max_iter=50,
        w=0.5,
        c1=0.5,
        c2=0.5,
    ):
        self.objective_function = objective_function
        self.bounds = np.array(
            bounds
        )  # [[min_h1, max_h1], [min_h2, max_h2], ...]
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient

        self.dimensions = len(bounds)
        self.particles = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (num_particles, self.dimensions),
        )
        self.velocities = np.zeros((num_particles, self.dimensions))

        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.array(
            [float("inf")] * num_particles
        )
        self.global_best_position = None
        self.global_best_score = float("inf")

    def optimize(self, data):
        # Initial evaluation of particles
        for i in range(self.num_particles):
            # Pass only the particle's position (hyperparameters) to the objective function
            score = self.objective_function(self.particles[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[
                    i
                ].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()

        # PSO iterations
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dimensions)
                r2 = np.random.rand(self.dimensions)

                # Update velocity
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1
                    * r1
                    * (
                        self.personal_best_positions[i]
                        - self.particles[i]
                    )
                    + self.c2
                    * r2
                    * (self.global_best_position - self.particles[i])
                )

                # Update position and apply bounds
                self.particles[i] += self.velocities[i]
                for d in range(self.dimensions):
                    self.particles[i, d] = np.clip(
                        self.particles[i, d],
                        self.bounds[d, 0],
                        self.bounds[d, 1],
                    )

                # Evaluate new position
                # Pass only the particle's new position (hyperparameters) to the objective function
                score = self.objective_function(self.particles[i])
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[
                        i
                    ].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[
                        i
                    ].copy()

            print(
                f"Iteration {iteration + 1}/{self.max_iter}, Best NLL: {self.global_best_score:.4f}, Best Hyperparams: {self.global_best_position.round()}"
            )

        return self.global_best_position, self.global_best_score


# %%
data_dim = 2
num_samples = 1000
mean1 = torch.tensor([2.0, 2.0])
cov1 = torch.eye(data_dim) * 0.5
dist1 = MultivariateNormal(mean1, cov1)

mean2 = torch.tensor([-2.0, -2.0])
cov2 = torch.eye(data_dim) * 0.8
dist2 = MultivariateNormal(mean2, cov2)

data = torch.cat([
    dist1.sample([num_samples // 2]),
    dist2.sample([num_samples // 2]),
])
print(f"Generated data shape: {data.shape}")

# Define hyperparameter search space for the PSO
# [num_layers_min, num_layers_max], [hidden_features_min, hidden_features_max]
hyperparam_bounds = [
    [1, 5],  # num_layers (integer)
    [32, 128],  # hidden_features (integer)
]

# Initialize and run PSO
pso = BasicPSO(
    objective_function=lambda hp: evaluate_flow_performance(
        hp, data
    ),
    bounds=hyperparam_bounds,
    num_particles=5,  # Small number for quick demo
    max_iter=10,  # Small number for quick demo
)

best_hyperparams, best_nll = pso.optimize(data)

print("\n--- PSO Optimization Complete ---")
print(
    f"Optimal Normalizing Flow Hyperparameters: {best_hyperparams.round().astype(int)}"
)
print(f"Achieved NLL: {best_nll:.4f}")

# You can now train the final model with these optimal hyperparameters
final_model = create_normalizing_flow_model(
    num_layers=int(best_hyperparams[0]),
    hidden_features=int(best_hyperparams[1]),
    input_dim=data_dim,
)
# Further training or deployment of final_model
print(
    "\nFinal model created with optimized hyperparameters. Ready for further training or sampling."
)
# %% [markdown]
"""# How this Prototype Works:

#    SimplifiedSelfAttention:
        This module simulates the output of a single attention head from a Transformer (like BERT)
        It takes dummy input embeddings (x) and applies linear transformations for Query, Key, and Value.
        It calculates attention scores and applies softmax to get attn_weights.
        Crucially, it returns these attn_weights (flattened) as the initial state for our ODE. This state represents the "attention configuration" at conceptual time t=0.

#    AttentionODE (The Velocity Field):
        This is the core of the ODE. It's a small neural network (self.net).
        Its forward method takes two arguments: t (the current conceptual time) and attention_state (the current flattened attention weights).
        It concatenates t with attention_state and feeds it through its network.
        The output is the predicted derivative (or velocity) of the attention_state with respect to t. This tells the ODE solver how the attention state should change instantaneously.

#    ODEAttentionModel:
        This module orchestrates the process.
        It takes an initial_attention_generator (our SimplifiedSelfAttention) to get the starting point.
        It instantiates the AttentionODE (self.ode_func).
        In its forward method, it calls torchdiffeq.odeint.
            odeint is the ODE solver.
            It takes self.ode_func (our velocity field), initial_attention_state, and t_span (the conceptual time interval, e.g., [0.0, 1.0]).
            odeint numerically integrates the ODE, starting from initial_attention_state at t_span[0], and returns the attention_state at t_span[1] (and any other points in t_span).

#    Main Execution and Training:
        We create dummy input embeddings to simulate token representations.
        We define a t_span from 0.0 to 1.0.
        We define a target_attention (e.g., a uniform distribution across all attention connections). This is our arbitrary goal for the attention to evolve towards.
        A simplified training loop uses Adam optimizer to minimize the MSELoss between the evolved_attention_state (output of the ODE) and our target_attention. This trains the weights of the AttentionODE's neural network.

# What it Demonstrates:

    Continuous Evolution: Instead of discrete layers, the attention state evolves continuously over a conceptual "time" axis.
    Neural Network as Velocity Field: The AttentionODE network learns the "rules" of this continuous evolution.
    Flexibility: By changing the target_attention and the loss function, you could train this system to make attention sparse, focus on specific tokens, or exhibit other desired dynamic behaviors.
    Trajectory Visualization: The example shows how you can query intermediate states along the ODE trajectory, allowing you to "see" how the attention evolves step-by-step.

This prototype provides a foundational understanding of how ODEs can be used to model the continuous dynamics of internal representations like attention weights in a neural network, opening doors for more advanced research into their behavior and control.
"""
# %%
class SimplifiedSelfAttention(nn.Module):
    """
    A very simplified self-attention mechanism to generate initial attention weights.
    This is NOT a full BERT attention head, just a way to get a matrix of 'attention-like' values.
    """

    def __init__(self, embed_dim, num_heads=1, seq_len=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len

        # Linear layers for Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection (optional, but common in Transformers)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Calculate attention scores (Q * K^T / sqrt(head_dim))
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )

        # Apply softmax to get attention probabilities (distribution over keys)
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # Shape: (B, H, S, S)

        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and project back to original embedding dimension
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        # For this prototype, we'll return the attention weights themselves as the state to evolve.
        # We'll flatten them for simplicity to be a 1D vector per batch item for the ODE.
        # Shape: (batch_size, num_heads * seq_len * seq_len)
        return attn_weights.view(batch_size, -1)
# %%

# --- 2. ODE Function (Velocity Field for Attention State) ---
class AttentionODE(nn.Module):
    """
    This defines the velocity field v_theta(attention_state, t) for the ODE.
    It takes the current attention state (flattened attention weights) and conceptual time 't',
    and outputs the predicted derivative (velocity) of the attention state.
    """

    def __init__(self, attention_state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                attention_state_dim + 1, hidden_dim
            ),  # +1 for time 't'
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, attention_state_dim
            ),  # Output is the velocity of the attention state
        )

    def forward(self, t, attention_state):
        """
        t: scalar tensor representing conceptual time.
        attention_state: tensor of shape (batch_size, attention_state_dim)
        """
        # Concatenate time 't' to the attention state.
        # t needs to be expanded to match batch_size
        t_expanded = t.expand(attention_state.shape[0], 1)
        input_tensor = torch.cat([attention_state, t_expanded], dim=1)
        return self.net(input_tensor)

# %%
# --- 3. ODE Attention Model (Wraps the ODE and Handles Integration) ---
class ODEAttentionModel(nn.Module):
    """
    This model initializes an attention state and evolves it using an ODE solver.
    """

    def __init__(
        self,
        initial_attention_generator,
        attention_state_dim,
        ode_hidden_dim,
    ):
        super().__init__()
        self.initial_attention_generator = initial_attention_generator
        self.ode_func = AttentionODE(
            attention_state_dim, ode_hidden_dim
        )

    def forward(self, x, t_span):
        """
        x: Input tensor for the initial attention generator (e.g., token embeddings).
        t_span: A tensor of conceptual time points (e.g., torch.tensor([0.0, 1.0])).
                The ODE will be integrated over this span.
        """
        # Get the initial attention state from the generator
        initial_attention_state = self.initial_attention_generator(x)

        # Integrate the ODE to get the evolved attention states
        # odeint returns solutions at each time point in t_span
        # result_attention_states shape: (len(t_span), batch_size, attention_state_dim)
        result_attention_states = odeint(
            self.ode_func,
            initial_attention_state,
            t_span,
            method="dopri5",
        )

        # We are interested in the final evolved attention state (at t_span[-1])
        final_attention_state = result_attention_states[-1]
        return final_attention_state


# %%
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters for our simplified setup
BATCH_SIZE = 4
SEQ_LEN = 5  # Number of tokens in a sequence
EMBED_DIM = 64  # Embedding dimension for tokens
NUM_HEADS = 1  # Simplified to one head for easier visualization
ODE_HIDDEN_DIM = 128

# Calculate the dimension of the flattened attention state
ATTENTION_STATE_DIM = NUM_HEADS * SEQ_LEN * SEQ_LEN

# --- 1. Instantiate the components ---
# This generates the initial attention state (e.g., from BERT's first layer)
initial_attention_generator = SimplifiedSelfAttention(
    EMBED_DIM, NUM_HEADS, SEQ_LEN
).to(device)

# This is our ODE-based model that evolves the attention state
ode_attention_model = ODEAttentionModel(
    initial_attention_generator=initial_attention_generator,
    attention_state_dim=ATTENTION_STATE_DIM,
    ode_hidden_dim=ODE_HIDDEN_DIM,
).to(device)

# --- 2. Create Dummy Input Data (e.g., token embeddings for a batch) ---
dummy_input_embeddings = torch.randn(
    BATCH_SIZE, SEQ_LEN, EMBED_DIM
).to(device)

# --- 3. Define Conceptual Time Span ---
# We want to evolve from t=0.0 to t=1.0
# The ODE solver will take intermediate steps.
t_span = torch.tensor([0.0, 1.0]).to(device)

# --- 4. Define a Placeholder Training Objective ---
# This is purely illustrative. A real objective would be task-specific.
# Example: Try to make the final attention state close to a uniform distribution
# or a sparse distribution.
target_attention_uniform = torch.ones(ATTENTION_STATE_DIM).to(
    device
) / (SEQ_LEN * SEQ_LEN)
target_attention_uniform = target_attention_uniform.repeat(
    BATCH_SIZE, 1
)

# Or a sparse target (e.g., only diagonal elements are 1, others 0)
target_attention_sparse = torch.zeros(ATTENTION_STATE_DIM).to(device)
# Reshape to (SEQ_LEN, SEQ_LEN) to set diagonal
sparse_matrix = torch.eye(SEQ_LEN).view(-1).to(device)
target_attention_sparse = sparse_matrix.repeat(BATCH_SIZE, 1)
# Let's use the uniform target for this demo
target_attention = target_attention_uniform
# --- 5. Training Loop (Simplified) ---
optimizer = torch.optim.Adam(
    ode_attention_model.parameters(), lr=1e-3
)
num_training_steps = 100  # Keep short for demo

print("\nStarting conceptual training loop...")
for step in range(num_training_steps):
    optimizer.zero_grad()

    # Forward pass: Evolve the attention state
    evolved_attention_state = ode_attention_model(
        dummy_input_embeddings, t_span
    )

    # Calculate loss (e.g., Mean Squared Error to target)
    loss = F.mse_loss(evolved_attention_state, target_attention)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        print(
            f"Step {step + 1}/{num_training_steps}, Loss: {loss.item():.4f}"
        )

print("\nConceptual training finished.")
# %%
# --- 6. Demonstrate Evolution ---
print("\n--- Demonstrating Attention Evolution ---")
# Get initial attention state
with torch.no_grad():
    initial_attention = initial_attention_generator(
        dummy_input_embeddings
    )
    print(
        f"\nInitial Attention State (first batch item, reshaped to {SEQ_LEN}x{SEQ_LEN}):"
    )
    print(
        initial_attention[0]
        .view(SEQ_LEN, SEQ_LEN)
        .cpu()
        .numpy()
        .round(3)
    )

    # Get evolved attention state after training
    evolved_attention = ode_attention_model(
        dummy_input_embeddings, t_span
    )
    print(
        f"\nEvolved Attention State (first batch item, reshaped to {SEQ_LEN}x{SEQ_LEN}):"
    )
    print(
        evolved_attention[0]
        .view(SEQ_LEN, SEQ_LEN)
        .cpu()
        .numpy()
        .round(3)
    )

    print(
        f"\nTarget Attention State (first batch item, reshaped to {SEQ_LEN}x{SEQ_LEN}):"
    )
    print(
        target_attention[0]
        .view(SEQ_LEN, SEQ_LEN)
        .cpu()
        .numpy()
        .round(3)
    )

    # You can also get intermediate states by changing t_span
    t_intermediate = torch.linspace(0.0, 1.0, 5).to(
        device
    )  # 5 points from 0 to 1
    intermediate_states = odeint(
        ode_attention_model.ode_func,
        initial_attention,
        t_intermediate,
        method="dopri5",
    )

    print(
        "\n--- Intermediate Attention States during Evolution (first batch item) ---"
    )
    for i, t_val in enumerate(t_intermediate):
        print(f"Time t={t_val.item():.2f}:")
        print(
            intermediate_states[i, 0]
            .view(SEQ_LEN, SEQ_LEN)
            .cpu()
            .numpy()
            .round(3)
        )
        print("-" * 20)

# %%
dataset = load_dataset(
    "code_x_glue_cc_code_completion_token", "python"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
TOK = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# OPTIMIZED: Reduced sequence length for much smaller attention matrices
MAX_LEN = 32  # Reduced from 128 to 32 (attention dims: 32x32=1024 vs 128x128=16384)


def encode(example):
    text = " ".join(example["code"])
    enc = TOK(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
        padding=False,
    )
    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
    }


# OPTIMIZED: Use even smaller subset for faster iteration
num_total_examples_train = len(dataset["train"])
num_examples_to_use = min(5000, int(num_total_examples_train * 0.01))  # 1% or 1000 max

if num_examples_to_use == 0:
    num_examples_to_use = min(100, num_total_examples_train)

print(f"Using {num_examples_to_use} examples for efficient training.")

original_columns = dataset["train"].column_names
code_ds = (
    dataset["train"]
    .select(range(num_examples_to_use))
    .map(encode, remove_columns=original_columns, batched=False)
)
code_ds.set_format("torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorWithPadding(
    tokenizer=TOK,
    padding="max_length",
    max_length=MAX_LEN,
)

# OPTIMIZED: Smaller batch size for memory efficiency
loader = DataLoader(
    code_ds,
    batch_size=4,  # Reduced from 8 to 4
    shuffle=True,
    drop_last=True,
    collate_fn=data_collator,
)

print("Optimized dataset preparation complete.")

# %%
CODEBERT = RobertaModel.from_pretrained(
    "microsoft/codebert-base", output_attentions=True
)
CODEBERT = CODEBERT.to(DEVICE).eval()

# OPTIMIZED: Use gradient checkpointing to save memory
CODEBERT.gradient_checkpointing_enable()

EARLY_L, LATE_L = 0, 5  # OPTIMIZED: Reduced layer gap from 11 to 5


@torch.no_grad()
def get_attention(ids, mask):
    # OPTIMIZED: Use autocast for mixed precision to save memory
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        out = CODEBERT(input_ids=ids, attention_mask=mask)
        # Only use first head to reduce complexity
        A0 = out.attentions[EARLY_L][:, 0]  # (B,S,S)
        AT = out.attentions[LATE_L][:, 0]
        return A0, AT


class OptimizedAttentionODE(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        # OPTIMIZED: Smaller, more efficient network
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.GELU(),  # More efficient than ReLU
            nn.Dropout(0.1),  # Add regularization
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, dim),
        )

    def forward(self, t, a):
        tvec = t.expand(a.size(0), 1)
        return self.net(torch.cat([a, tvec], dim=1))


# OPTIMIZED: More efficient ODE integration
ODE_STEPS = torch.tensor([0.0, 1.0], device=DEVICE)

ATTN_DIM = MAX_LEN * MAX_LEN  # Now 32*32=1024 instead of 128*128=16384
ode_func = OptimizedAttentionODE(dim=ATTN_DIM).to(DEVICE)
optim_ode = torch.optim.AdamW(ode_func.parameters(), lr=1e-4, weight_decay=1e-5)

# OPTIMIZED: Use mixed precision training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

print("Training optimized ODE model...")
for epoch in range(1):
    pbar = tqdm.tqdm(loader, desc="ODE-train")
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        # OPTIMIZED: Use mixed precision
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            A0, AT = get_attention(ids, mask)
            A0f = A0.flatten(1)
            ATf = AT.flatten(1)
            # OPTIMIZED: Use faster ODE method with lower tolerance
            pred = odeint(
                ode_func, 
                A0f, 
                ODE_STEPS, 
                method="euler",  # Faster than dopri5
                options={"step_size": 0.1}  # Fixed step size for speed
            )[-1]
            loss = F.mse_loss(pred, ATf)
        optim_ode.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim_ode)
            scaler.update()
        else:
            loss.backward()
            optim_ode.step()
        pbar.set_postfix({"mse": f"{loss.item():.3e}"})


@torch.no_grad()
def evolve_attention(ids, mask):
    A0, _ = get_attention(ids, mask)
    A0f = A0.flatten(1)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        AT_pred = odeint(
            ode_func, 
            A0f, 
            ODE_STEPS, 
            method="euler",
            options={"step_size": 0.1},
            atol=1e-2,  # Relaxed tolerance
            rtol=1e-2
        )[-1]
    return AT_pred.view(ids.size(0), MAX_LEN, MAX_LEN)


class OptimizedPositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):  # Reduced max_len
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class OptimizedAttnGuidedDecoder(nn.Module):
    def __init__(self, vocab, d_model=256, layers=1):  # OPTIMIZED: Smaller model
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.posenc = OptimizedPositionalEncoding(d_model)
        # OPTIMIZED: Simpler decoder architecture
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, 
            nhead=4,  # Reduced from 8
            dim_feedforward=512,  # Reduced from default 2048
            dropout=0.1,
            batch_first=True
        )
        self.proj_attn = nn.Linear(MAX_LEN, d_model)  # Project attention to model dim
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, tgt_ids, evolved_attn):
        # OPTIMIZED: More efficient attention handling
        B, S, _ = evolved_attn.shape
        mem = self.proj_attn(evolved_attn)  # (B, S, d_model)
        tgt = self.posenc(self.embed(tgt_ids))
        # Create causal mask
        tgt_mask = torch.triu(torch.ones(S, S, device=tgt_ids.device), diagonal=1).bool()
        out = self.decoder_layer(tgt, mem, tgt_mask=tgt_mask)
        return self.proj(out)


vocab = len(TOK)
decoder = OptimizedAttnGuidedDecoder(vocab).to(DEVICE)
optim_dec = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-5)

print("Training optimized decoder...")
for epoch in range(1):
    pbar = tqdm.tqdm(loader, desc="DEC-train")
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            evolved = evolve_attention(ids, mask)
            logits = decoder(ids, evolved)  # Use same ids for simplicity
            loss = F.cross_entropy(logits.view(-1, vocab), ids.view(-1), ignore_index=TOK.pad_token_id)

        optim_dec.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim_dec)
            scaler.update()
        else:
            loss.backward()
        pbar.set_postfix({"ce": f"{loss.item():.3f}"})

print("\nOptimized demonstration complete!")
print("Memory usage optimizations:")
print("- Sequence length: 128 → 32 (16x less attention parameters)")
print("- Batch size: 8 → 4")
print("- ODE hidden dim: 1024 → 256")
print("- Decoder layers: 2 → 1")
print("- Mixed precision training enabled")
print("- Efficient ODE solver (euler vs dopri5)")
# %%
# === ANALYSIS AND RESULTS ===
print("\n" + "="*60)
print("           TRAINING ANALYSIS & RESULTS")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import cosine_similarity

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# %%
# 1. ATTENTION EVOLUTION ANALYSIS
print("\n1. ATTENTION EVOLUTION ANALYSIS")
print("-" * 40)

# Get a sample batch for analysis
sample_batch = next(iter(loader))
sample_ids = sample_batch["input_ids"][:2].to(DEVICE)  # Take first 2 samples
sample_mask = sample_batch["attention_mask"][:2].to(DEVICE)

# Get initial and target attention from CodeBERT
with torch.no_grad():
    A0_sample, AT_sample = get_attention(sample_ids, sample_mask)
    # Get our model's evolved attention
    evolved_sample = evolve_attention(sample_ids, sample_mask)
    print(f"Sample shape: {sample_ids.shape}")
    print(f"Attention matrices shape: {A0_sample.shape}")
    # Calculate similarity metrics
    A0_flat = A0_sample.flatten(1)
    AT_flat = AT_sample.flatten(1) 
    evolved_flat = evolved_sample.flatten(1)
    # Cosine similarities
    cos_sim_original = cosine_similarity(A0_flat, AT_flat, dim=1).mean()
    cos_sim_evolved = cosine_similarity(A0_flat, evolved_flat, dim=1).mean()
    cos_sim_target = cosine_similarity(evolved_flat, AT_flat, dim=1).mean()
    print(f"Cosine Similarity - Initial to Target (CodeBERT): {cos_sim_original:.4f}")
    print(f"Cosine Similarity - Initial to Evolved (Our Model): {cos_sim_evolved:.4f}")
    print(f"Cosine Similarity - Evolved to Target: {cos_sim_target:.4f}")
    # MSE losses
    mse_original = F.mse_loss(A0_flat, AT_flat).item()
    mse_evolved = F.mse_loss(evolved_flat, AT_flat).item()
    print(f"MSE - Initial to Target: {mse_original:.6f}")
    print(f"MSE - Evolved to Target: {mse_evolved:.6f}")
    print(f"Improvement: {((mse_original - mse_evolved) / mse_original * 100):.2f}%")

# %%
# 2. ATTENTION PATTERN VISUALIZATION
print("\n2. ATTENTION PATTERN VISUALIZATION")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sample_idx = 0  # Visualize first sample

# Convert to numpy for plotting
A0_np = A0_sample[sample_idx].cpu().numpy()
AT_np = AT_sample[sample_idx].cpu().numpy()
evolved_np = evolved_sample[sample_idx].cpu().numpy()

# Plot attention matrices
im1 = axes[0, 0].imshow(A0_np, cmap='Blues', vmin=0, vmax=0.5)
axes[0, 0].set_title('Initial Attention (Layer 0)')
axes[0, 0].set_xlabel('Key Position')
axes[0, 0].set_ylabel('Query Position')
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(AT_np, cmap='Blues', vmin=0, vmax=0.5)
axes[0, 1].set_title('Target Attention (Layer 5)')
axes[0, 1].set_xlabel('Key Position')
axes[0, 1].set_ylabel('Query Position')
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[0, 2].imshow(evolved_np, cmap='Blues', vmin=0, vmax=0.5)
axes[0, 2].set_title('Evolved Attention (Our Model)')
axes[0, 2].set_xlabel('Key Position')
axes[0, 2].set_ylabel('Query Position')
plt.colorbar(im3, ax=axes[0, 2])

# Plot difference maps
diff_original = np.abs(A0_np - AT_np)
diff_evolved = np.abs(evolved_np - AT_np)

im4 = axes[1, 0].imshow(diff_original, cmap='Reds', vmin=0, vmax=0.3)
axes[1, 0].set_title('|Initial - Target|')
axes[1, 0].set_xlabel('Key Position')
axes[1, 0].set_ylabel('Query Position')
plt.colorbar(im4, ax=axes[1, 0])

im5 = axes[1, 1].imshow(diff_evolved, cmap='Reds', vmin=0, vmax=0.3)
axes[1, 1].set_title('|Evolved - Target|')
axes[1, 1].set_xlabel('Key Position')
axes[1, 1].set_ylabel('Query Position')
plt.colorbar(im5, ax=axes[1, 1])

# Improvement map
improvement = diff_original - diff_evolved
im6 = axes[1, 2].imshow(improvement, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
axes[1, 2].set_title('Improvement Map')
axes[1, 2].set_xlabel('Key Position')
axes[1, 2].set_ylabel('Query Position')
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.show()

# %%
# 3. ATTENTION STATISTICS ANALYSIS
print("\n3. ATTENTION STATISTICS ANALYSIS")
print("-" * 40)

# Calculate attention statistics
def analyze_attention_stats(attention_matrix, name):
    """Analyze statistical properties of attention matrices"""
    attn_flat = attention_matrix.flatten()
    stats = {
        'mean': attn_flat.mean().item(),
        'std': attn_flat.std().item(),
        'max': attn_flat.max().item(),
        'min': attn_flat.min().item(),
        'entropy': -(attn_flat * torch.log(attn_flat + 1e-8)).sum().item(),
        'sparsity': (attn_flat < 0.01).float().mean().item()
    }
    print(f"{name}:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    return stats

stats_initial = analyze_attention_stats(A0_sample, "Initial Attention")
stats_target = analyze_attention_stats(AT_sample, "Target Attention") 
stats_evolved = analyze_attention_stats(evolved_sample, "Evolved Attention")

# %%
# 4. ODE TRAJECTORY ANALYSIS
print("\n4. ODE TRAJECTORY ANALYSIS")
print("-" * 40)

# Analyze the ODE trajectory at multiple time points
time_points = torch.linspace(0.0, 1.0, 6).to(DEVICE)

with torch.no_grad():
    A0_traj = get_attention(sample_ids[:1], sample_mask[:1])[0]
    A0_flat_traj = A0_traj.flatten(1)
    # Get trajectory
    trajectory = odeint(
        ode_func, 
        A0_flat_traj, 
        time_points, 
        method="euler",
        options={"step_size": 0.05}
    )
    print("Trajectory Analysis:")
    print(f"Time points: {time_points.cpu().numpy()}")
    # Calculate changes at each time point
    for i, t in enumerate(time_points):
        if i > 0:
            prev_state = trajectory[i-1]
            curr_state = trajectory[i]
            change = F.mse_loss(curr_state, prev_state).item()
            print(f"t={t.item():.2f}: MSE change from previous = {change:.6f}")

# Plot trajectory evolution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
trajectory_norms = []
for i in range(len(time_points)):
    state = trajectory[i, 0].cpu().numpy()  # First sample
    norm = np.linalg.norm(state)
    trajectory_norms.append(norm)

ax.plot(time_points.cpu().numpy(), trajectory_norms, 'o-', linewidth=2, markersize=6)
ax.set_xlabel('Time')
ax.set_ylabel('Attention State L2 Norm')
ax.set_title('ODE Trajectory: Attention State Evolution')
ax.grid(True, alpha=0.3)
plt.show()

# %%
# 5. MODEL PERFORMANCE METRICS
print("\n5. MODEL PERFORMANCE METRICS")
print("-" * 40)

# Test on a larger sample for more robust metrics
test_samples = 10
total_mse_improvement = 0
total_cos_sim_improvement = 0

with torch.no_grad():
    for i in range(0, min(test_samples, len(code_ds)), 2):
        # Get batch
        batch_start = i
        batch_end = min(i + 2, len(code_ds))
        test_batch = [code_ds[j] for j in range(batch_start, batch_end)]
        # Create proper batch
        collated = data_collator(test_batch)
        test_ids = collated["input_ids"].to(DEVICE)
        test_mask = collated["attention_mask"].to(DEVICE)
        # Get attentions
        A0_test, AT_test = get_attention(test_ids, test_mask)
        evolved_test = evolve_attention(test_ids, test_mask)
        # Calculate metrics
        A0_flat_test = A0_test.flatten(1)
        AT_flat_test = AT_test.flatten(1)
        evolved_flat_test = evolved_test.flatten(1)
        # MSE improvement
        mse_original_test = F.mse_loss(A0_flat_test, AT_flat_test).item()
        mse_evolved_test = F.mse_loss(evolved_flat_test, AT_flat_test).item()
        mse_improvement = (mse_original_test - mse_evolved_test) / mse_original_test
        total_mse_improvement += mse_improvement
        # Cosine similarity improvement
        cos_original_test = cosine_similarity(A0_flat_test, AT_flat_test, dim=1).mean().item()
        cos_evolved_test = cosine_similarity(evolved_flat_test, AT_flat_test, dim=1).mean().item()
        cos_improvement = cos_evolved_test - cos_original_test
        total_cos_sim_improvement += cos_improvement

avg_mse_improvement = total_mse_improvement / (test_samples // 2)
avg_cos_improvement = total_cos_sim_improvement / (test_samples // 2)

print(f"Average MSE Improvement: {avg_mse_improvement:.4f} ({avg_mse_improvement*100:.2f}%)")
print(f"Average Cosine Similarity Improvement: {avg_cos_improvement:.4f}")

# %%
# 6. COMPUTATIONAL EFFICIENCY ANALYSIS
print("\n6. COMPUTATIONAL EFFICIENCY ANALYSIS")
print("-" * 40)

import time
import psutil
import torch

# Memory usage analysis
if torch.cuda.is_available():
    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
    print(f"GPU Memory Allocated: {gpu_memory:.2f} GB")
    print(f"GPU Memory Cached: {gpu_memory_cached:.2f} GB")

# CPU memory usage
cpu_memory = psutil.Process().memory_info().rss / 1024**3  # GB
print(f"CPU Memory Usage: {cpu_memory:.2f} GB")

# Timing analysis
print("\nTiming Analysis:")

# Time ODE forward pass
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        evolved_timing = evolve_attention(sample_ids, sample_mask)
ode_time = (time.time() - start_time) / 10

# Time CodeBERT forward pass for comparison
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = get_attention(sample_ids, sample_mask)
codebert_time = (time.time() - start_time) / 10

print(f"Average ODE Evolution Time: {ode_time:.4f} seconds")
print(f"Average CodeBERT Attention Time: {codebert_time:.4f} seconds")
print(f"Speed Ratio (ODE/CodeBERT): {ode_time/codebert_time:.2f}x")

