# %%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_california_housing
from datasets import load_dataset

nltk.download("wordnet")


# %% ======================================================
def generate_brownian_bridge(
    tau_start, tau_end, x_start, x_final, num_steps, sigma_scale=1.0
):
    """
    Generates a 1D Brownian bridge.
    This represents a sample path from the Euclidean path integral for a free particle.
    The path integral for a free particle is Gaussian.
    (Same implementation as before)
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    tau_coords = np.linspace(tau_start, tau_end, num_steps + 1)

    if num_steps == 0:  # Edge case
        return tau_coords, np.array(
            [x_start]
            if x_start == x_final and tau_start == tau_end
            else [x_start, x_final]
        )

    dt = (tau_end - tau_start) / num_steps

    if num_steps > 0:
        dW_scaled = (
            sigma_scale
            * np.sqrt(dt)
            * np.random.normal(0, 1, num_steps)
        )
    else:
        dW_scaled = np.array([])

    W_scaled = np.zeros(num_steps + 1)
    W_scaled[1:] = np.cumsum(dW_scaled)

    T_prime = tau_end - tau_start
    t_prime_coords = tau_coords - tau_start

    if T_prime == 0:
        if x_start == x_final:
            return tau_coords, np.full_like(tau_coords, x_start)
        else:
            return tau_coords, np.linspace(
                x_start, x_final, num_steps + 1
            )

    W_scaled_T_prime = W_scaled[-1]
    brownian_bridge_component = (
        W_scaled - (t_prime_coords / T_prime) * W_scaled_T_prime
    )
    classical_path_segment = x_start + (x_final - x_start) * (
        t_prime_coords / T_prime
    )
    x_coords = classical_path_segment + brownian_bridge_component

    x_coords[0] = x_start
    x_coords[-1] = x_final

    return tau_coords, x_coords


# %% ======================================================
# --- Parameters for visualization ---
tau_initial = 0.0
tau_final = 1.0
x_initial = 0.0
x_final = 0.5

num_time_steps = 100
num_paths_to_show = 15

# Define two different fluctuation scales
fluctuation_scale_large = 0.3  # Wider fluctuations
fluctuation_scale_small = 0.05  # Narrower fluctuations

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

scales_and_titles = [
    (
        fluctuation_scale_large,
        f"Larger Fluctuations (Scale: {fluctuation_scale_large})",
    ),
    (
        fluctuation_scale_small,
        f"Smaller Fluctuations (Scale: {fluctuation_scale_small})",
    ),
]
for ax_idx, (current_scale, title_text) in enumerate(
    scales_and_titles
):
    ax = axes[ax_idx]
    # 1. Plot the classical path
    classical_tau_coords = np.array([tau_initial, tau_final])
    classical_x_coords = np.array([x_initial, x_final])
    ax.plot(
        classical_tau_coords,
        classical_x_coords,
        "r-",
        lw=2,
        label="Classical Path",
        zorder=10,
    )

    # 2. Plot several sample quantum paths
    for i in range(num_paths_to_show):
        tau_path, x_path = generate_brownian_bridge(
            tau_initial,
            tau_final,
            x_initial,
            x_final,
            num_time_steps,
            sigma_scale=current_scale,
        )
        ax.plot(
            tau_path,
            x_path,
            "b-",
            lw=0.7,
            alpha=0.5,
            label="Sample Quantum Path" if i == 0 else None,
        )

    # 3. Highlight start and end points
    ax.plot(
        tau_initial,
        x_initial,
        "go",
        markersize=8,
        label=f"Start",
        zorder=11,
    )
    ax.plot(
        tau_final,
        x_final,
        "mo",
        markersize=8,
        label=f"End",
        zorder=11,
    )

    ax.set_xlabel("Imaginary Time (τ)")
    if ax_idx == 0:
        ax.set_ylabel("Position (x)")
    ax.set_title(title_text)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax.grid(True)
    xlim_padding = (
        0.05 * (tau_final - tau_initial)
        if (tau_final - tau_initial) > 0
        else 0.05
    )
    ax.set_xlim(tau_initial - xlim_padding, tau_final + xlim_padding)

fig.suptitle(
    "Effect of Fluctuation Scale on Gaussian Path Integral Samples (Free Particle)",
    fontsize=16,
)
plt.tight_layout(
    rect=[0, 0.03, 1, 0.95]
)  # Adjust layout for suptitle and figtext
fig.text(
    0.5,
    0.005,
    "The 'width' of the quantum paths is controlled by a scale factor (analogous to √(ħ/m) for a particle).\n"
    "Smaller scale implies a more 'classical-like' system, where paths cluster tightly around the classical trajectory.",
    ha="center",
    va="bottom",
    fontsize=10,
    bbox={"facecolor": "lightgray", "alpha": 0.7, "pad": 5},
)
plt.show()
plt.show()
# %% [markdown]
# --- Gaussian Path Integral Fluctuation Control ---
#
# The plots demonstrate how changing the 'sigma_scale' (quantum_fluctuation_scale) affects the spread of paths.
#
# - With scale = {fluctuation_scale_large:.2f}, paths fluctuate more widely.
# - With scale = {fluctuation_scale_small:.2f}, paths are much more concentrated around the classical (red) line.
#
# This scale factor is analogous to physical parameters:
#
#   - For a particle: Proportional to sqrt(ħ/m). A heavier particle (larger m) or smaller ħ would lead to narrower paths.
#   - For EM field modes (QHOs): Related to properties of the mode. The path integral for these modes is also Gaussian.
#
# Important: For systems like free particles or QHOs (which model EM field modes for blackbody radiation),
# the Gaussian path integral is exact. The 'width' is a fundamental quantum property, not an artifact of an approximation scheme for these specific systems.


# %%
def generate_minkowski_fluctuating_path(
    event_start, event_end, num_steps, spatial_fluctuation_scale
):
    """
    Generates a 'fluctuating path' in (1+1)D Minkowski-like spacetime between two events.
    The 'time' coordinate progresses monotonically. Fluctuations are added to the spatial coordinate.
    This is an analogy to path integral paths, not a rigorous Minkowski path integral.

    Args:
        event_start (tuple): (t_start, x_start) for the parent word vector.
        event_end (tuple): (t_end, x_end) for the child word vector.
        num_steps (int): Number of segments in the path.
        spatial_fluctuation_scale (float): Scale of random spatial deviations.

    Returns:
        tuple: (t_coords, x_coords) for the path.
    """
    t_start, x_start = event_start
    t_end, x_end = event_end

    if t_end <= t_start:
        # For this semantic interpretation, child must be in the future of parent.
        # The paper's algorithm enforces t_child > t_parent.
        # We can slightly adjust if they are too close for num_steps > 0
        if num_steps > 0 and t_end == t_start:
            t_end += 1e-9  # ensure dt > 0 for calculation
        elif num_steps > 0 and t_end < t_start:
            raise ValueError(
                "Child event (t_end) must be later than parent event (t_start)."
            )

    t_coords = np.linspace(t_start, t_end, num_steps + 1)

    if num_steps == 0:
        return t_coords, np.array(
            [x_start, x_end] if t_start != t_end else [x_start]
        )

    # Classical path (straight line in space over the time interval)
    x_classical_path = np.linspace(x_start, x_end, num_steps + 1)

    # Generate spatial fluctuations (Brownian bridge like for the spatial component)
    # dt_segment is conceptual here for scaling noise, as 't' is our "time" axis
    dt_segment = (
        (t_end - t_start) / num_steps if num_steps > 0 else 1.0
    )

    # Generate random increments for spatial deviation
    # Scaled to be smaller for more steps over the same interval
    if num_steps > 0:
        # N(0,1) * scale * sqrt(dt_segment_normalized_to_1_for_total_interval)
        # This scaling is somewhat heuristic for visual effect.
        # A proper Wiener process in space parametrized by time 't' would have dW ~ sqrt(dt)
        spatial_increments = (
            spatial_fluctuation_scale
            * np.sqrt(dt_segment / (t_end - t_start + 1e-9))
            * np.random.normal(0, 1, num_steps)
        )
    else:
        spatial_increments = np.array([])

    spatial_deviations = np.zeros(num_steps + 1)
    spatial_deviations[1:] = np.cumsum(spatial_increments)

    # Make it a bridge: subtract the linear trend of the random walk
    if num_steps > 0:
        time_fractions = np.linspace(0, 1, num_steps + 1)
        spatial_bridge_component = (
            spatial_deviations
            - time_fractions * spatial_deviations[-1]
        )
    else:
        spatial_bridge_component = (
            np.array([0.0])
            if num_steps == 0
            else np.array([0.0, 0.0])
        )

    x_coords = x_classical_path + spatial_bridge_component

    # Ensure start and end spatial points are exact
    x_coords[0] = x_start
    x_coords[-1] = x_end

    return t_coords, x_coords


# --- Define Parent and Child "Word Vectors" (as events in Minkowski spacetime) ---
# Based on the paper, parent is in the past of the child.
# Let's use coordinates similar to those in the paper's Figure 2 (left panel, two nodes)
# Parent (e.g., a red dot lower down)
parent_event = (0.5, -0.2)  # (t_parent, x_parent)
# Child (e.g., a red dot higher up, connected to parent)
child_event = (1.0, -0.1)  # (t_child, x_child)

# The paper mentions near-null geodesics: dt^2 approx dx^2
dt = child_event[0] - parent_event[0]
dx = child_event[1] - parent_event[1]
print(
    f"Spacetime interval for classical path: dt={dt:.2f}, dx={dx:.2f}"
)
print(f"dt^2 = {dt**2:.3f}, dx^2 = {dx**2:.3f}")
if dt**2 < dx**2:
    print(
        "Warning: Classical path is spacelike! Child not in causal future of parent."
    )
    print(
        "This setup might not perfectly reflect the paper's causal embedding intention for the classical path."
    )
    print(
        "For the visualization, we will proceed, but keep this in mind."
    )
elif dt**2 == dx**2:
    print("Classical path is lightlike (null geodesic).")
else:
    print("Classical path is timelike.")


# --- Parameters for Visualization ---
num_path_segments = 50
num_sample_paths = 20
# This "perturbation_strength" controls the spatial width of the fluctuating paths
# Analogous to sqrt(ħ/m) or inverse of "stiffness"
perturbation_strength = 0.05  # Smaller for paths closer to classical

plt.figure(figsize=(10, 8))

# 1. Plot the "Classical Path" (Geodesic in Minkowski space)
# This is the direct connection as per the paper's embedding logic.
plt.plot(
    [parent_event[0], child_event[0]],
    [parent_event[1], child_event[1]],
    "r-",
    lw=2.5,
    label="Classical Path (Ideal Hierarchical Link)",
    zorder=10,
)

# 2. Plot sample "quantum fluctuating paths"
for i in range(num_sample_paths):
    t_path, x_path = generate_minkowski_fluctuating_path(
        parent_event,
        child_event,
        num_path_segments,
        spatial_fluctuation_scale=perturbation_strength,
    )
    plt.plot(
        t_path,
        x_path,
        "b-",
        lw=0.8,
        alpha=0.5,
        label="Sample Fluctuating Path" if i == 0 else None,
    )

# 3. Highlight Parent and Child events
plt.plot(
    parent_event[0],
    parent_event[1],
    "go",
    markersize=12,
    label='Parent Concept ("mammal")',
    zorder=11,
)
plt.plot(
    child_event[0],
    child_event[1],
    "mo",
    markersize=12,
    label='Child Concept ("dog")',
    zorder=11,
)

# 4. Light cones (optional, for context)
# Plot part of the future light cone of the parent
cone_t_parent = np.linspace(
    parent_event[0], child_event[0] * 1.1, 50
)  # Extend a bit beyond child time
plt.plot(
    cone_t_parent,
    parent_event[1] + (cone_t_parent - parent_event[0]),
    "k--",
    alpha=0.4,
    lw=1,
    label="Light Cone Boundary",
)
plt.plot(
    cone_t_parent,
    parent_event[1] - (cone_t_parent - parent_event[0]),
    "k--",
    alpha=0.4,
    lw=1,
)

# Plot part of the past light cone of the child
cone_t_child = np.linspace(
    child_event[0], parent_event[0] * 0.9, 50
)  # Extend a bit before parent time
plt.plot(
    cone_t_child,
    child_event[1] + (child_event[0] - cone_t_child),
    "k--",
    alpha=0.4,
    lw=1,
)
plt.plot(
    cone_t_child,
    child_event[1] - (child_event[0] - cone_t_child),
    "k--",
    alpha=0.4,
    lw=1,
)


plt.xlabel("Time Coordinate (t) in Semantic Spacetime")
plt.ylabel("Spatial Coordinate (x) in Semantic Spacetime")
plt.title(
    f"Path Integral Analogy for Parent-Child Word Vectors\nPerturbation Strength: {perturbation_strength}"
)

# Custom legend handling for unique labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="lower right")

plt.grid(True)
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.gca().set_aspect(
    "equal", adjustable="box"
)  # Makes light cones 45 degrees if scales are equal
plt.xlim(
    min(parent_event[0], child_event[0]) - 0.2,
    max(parent_event[0], child_event[0]) + 0.2,
)
plt.ylim(
    min(parent_event[1], child_event[1]) - 0.3,
    max(parent_event[1], child_event[1]) + 0.3,
)


plt.figtext(
    0.5,
    0.005,
    "Red: Ideal hierarchical link (classical path).\n"
    "Blue: Sample 'fluctuating paths' representing deviations or uncertainties (perturbations).\n"
    "Dashed: Light cone boundaries (causal limits). Child should be in parent's future light cone.",
    ha="center",
    va="bottom",
    fontsize=9,
    bbox={"facecolor": "lightgray", "alpha": 0.7, "pad": 5},
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
# %% ======================================================
# %%[markdown]
# --- Path Integral Analogy for Semantic Hierarchy ---
#
# This visualization is an *analogy* applying path integral concepts to the geometric embedding of meaning from the paper.
# It does NOT represent a formal calculation of a path integral for word vectors.
#
# Interpretation:
# - Parent ('mammal' at ({parent_event[0]:.2f}, {parent_event[1]:.2f})) and Child ('dog' at ({child_event[0]:.2f}, {child_event[1]:.2f})) concepts are events in a Minkowski-like spacetime.
# - The 'Classical Path' (Red Line) is the direct, ideal hierarchical connection, analogous to a geodesic. The paper aims for such paths to be near-null.
# - 'Sample Fluctuating Paths' (Blue Lines) represent a 'sum over histories' where the connection is not perfectly sharp. This could model:
#   - Imperfections or noise in the embedding process.
#   - The influence of other concepts slightly perturbing the ideal link.
#   - A hypothetical 'quantum-like' nature of semantic relationships where all consistent paths contribute.
# - Perturbation Strength ({perturbation_strength}): Controls how much the blue paths deviate. Smaller values mean the system is closer to the 'classical' ideal hierarchy.
# - Light Cones: Indicate causal boundaries. For a strict hierarchy, the child event should be within the future light cone of the parent event.

# %% ======================================================
# --- Simulation Parameters ---
# ======================================================
# Semantic Gravity
G_s = 0.01  # Semantic gravitational constant
m_p = 1.0  # Mass of parent concept
m_c = (
    0.1  # Mass of child concept (lighter, so it orbits more visibly)
)
epsilon_soft = (
    0.01  # Softening factor for gravity to avoid singularity
)

# Initial Conditions for the classical system
t_start = 0.5
t_end = 1.5  # Increased t_end to see more of the "orbit"
num_t_steps = (
    200  # Number of steps in semantic time for classical path
)

# Parent's mean classical path (e.g., moving slowly in x)
parent_x_start = -0.2
parent_vx_mean = 0.05  # Parent has a slight "drift" in x over time

# Child's initial state relative to parent at t_start
child_x_initial_offset = 0.1  # Initial spatial offset from parent
child_vx_initial_relative = (
    -0.2
)  # Initial spatial velocity relative to parent (to induce "orbit")

# Path Integral Fluctuation Parameters
num_path_segments_fluctuations = num_t_steps  # Match for simplicity
num_sample_paths = 20
perturbation_strength_parent = (
    0.02  # Fluctuation around parent's mean path
)
perturbation_strength_child = (
    0.03  # Fluctuation around child's gravitational path
)


# --- Helper function for Brownian Bridge (from previous examples) ---
def generate_brownian_bridge_component(num_steps, scale):
    """Generates a 1D Brownian bridge component starting and ending at 0."""
    if num_steps == 0:
        return np.array([0.0])

    dt_norm = (
        1.0 / num_steps
    )  # Normalized time step for bridge construction

    # Increments for a scaled Wiener process
    dW_scaled = (
        scale * np.sqrt(dt_norm) * np.random.normal(0, 1, num_steps)
    )

    W_scaled = np.zeros(num_steps + 1)
    W_scaled[1:] = np.cumsum(dW_scaled)  # W_scaled(0)=0

    # Construct the bridge component B(s) = W_scaled(s) - s * W_scaled(1) where s in [0,1]
    s_coords = np.linspace(0, 1, num_steps + 1)
    brownian_bridge_comp = W_scaled - s_coords * W_scaled[-1]
    return brownian_bridge_comp


# --- Simulate the Classical Orbital Path ---
t_coords_classical = np.linspace(t_start, t_end, num_t_steps + 1)
dt_sim = (t_end - t_start) / num_t_steps

# Parent's mean classical trajectory
parent_x_classical_mean = np.zeros_like(t_coords_classical)
parent_x_classical_mean[0] = parent_x_start
for i in range(num_t_steps):
    parent_x_classical_mean[i + 1] = (
        parent_x_classical_mean[i] + parent_vx_mean * dt_sim
    )

# Child's classical trajectory (orbiting parent)
child_x_classical = np.zeros_like(t_coords_classical)
child_vx_current_relative = (
    child_vx_initial_relative  # This is vx_child - vx_parent_mean
)
child_x_classical[0] = (
    parent_x_classical_mean[0] + child_x_initial_offset
)

for i in range(num_t_steps):
    # Current positions for force calculation
    xp_i = parent_x_classical_mean[i]
    xc_i = child_x_classical[i]

    # Spatial separation and distance for gravity
    r_x = xc_i - xp_i
    dist_sq_soft = r_x**2 + epsilon_soft**2

    # Gravitational force on child (1D)
    # F = -G * m1 * m2 / r^2 * sign(r)
    # Simplified: F_on_child = - G_s * m_p * m_c * r_x / (abs(r_x) * dist_sq_soft) -> more stable
    if abs(r_x) < 1e-9:  # Avoid division by zero if perfectly aligned
        force_g_x = 0
    else:
        force_g_x = (
            -G_s
            * m_p
            * m_c
            * r_x
            / (dist_sq_soft * np.sqrt(dist_sq_soft))
        )  # F = -GMm r_vec / |r|^3

    # Acceleration of child
    accel_c_x = force_g_x / m_c

    # Update child's velocity and position (Euler-Cromer)
    # Child's velocity is relative to the "lab frame", not the parent's mean velocity
    # For simplicity, let's track total velocity of child
    # Effective velocity of parent at step i
    # vx_parent_at_i = (parent_x_classical_mean[i+1 if i < num_t_steps-1 else i] - parent_x_classical_mean[i]) / dt_sim
    # if i == 0: # Initial total velocity for child
    #    child_vx_total_current = parent_vx_mean + child_vx_initial_relative

    # Let's simplify: child_vx is its absolute velocity in x
    if i == 0:
        # Initial total velocity of child. Assume parent starts with parent_vx_mean
        # child_vx_total = parent_vx_mean + child_vx_initial_relative
        # For a more stable orbit, let's define absolute initial velocity
        child_vx_total_current = (
            parent_vx_mean + child_vx_initial_relative
        )  # Initial guess

    child_vx_total_current = (
        child_vx_total_current + accel_c_x * dt_sim
    )
    child_x_classical[i + 1] = (
        child_x_classical[i] + child_vx_total_current * dt_sim
    )

print(f"Classical path calculation complete.")
print(
    f"Parent mean path starts at x={parent_x_classical_mean[0]:.3f}, ends at x={parent_x_classical_mean[-1]:.3f}"
)
print(
    f"Child classical path starts at x={child_x_classical[0]:.3f}, ends at x={child_x_classical[-1]:.3f}"
)

# %%
# --- Visualization ---
plt.figure(figsize=(12, 9))

# 1. Plot Classical "Orbital" Paths
plt.plot(
    t_coords_classical,
    parent_x_classical_mean,
    "r-",
    lw=2.5,
    label="Parent Mean Classical Path",
    zorder=10,
)
plt.plot(
    t_coords_classical,
    child_x_classical,
    "m-",
    lw=2.5,
    label="Child Gravitational Classical Path",
    zorder=10,
)

# 2. Plot sample "quantum fluctuating paths"
for i in range(num_sample_paths):
    # Fluctuations for parent
    parent_fluctuations = generate_brownian_bridge_component(
        num_t_steps, perturbation_strength_parent
    )
    parent_x_fluctuating = (
        parent_x_classical_mean + parent_fluctuations
    )

    # Fluctuations for child
    child_fluctuations = generate_brownian_bridge_component(
        num_t_steps, perturbation_strength_child
    )
    child_x_fluctuating = child_x_classical + child_fluctuations

    plt.plot(
        t_coords_classical,
        parent_x_fluctuating,
        "b-",
        lw=0.7,
        alpha=0.3,
        label="Parent Fluctuating Sample" if i == 0 else None,
    )
    plt.plot(
        t_coords_classical,
        child_x_fluctuating,
        "c-",
        lw=0.7,
        alpha=0.3,
        label="Child Fluctuating Sample" if i == 0 else None,
    )

# 3. Highlight Start and End Events for Classical Paths
plt.plot(
    t_coords_classical[0],
    parent_x_classical_mean[0],
    "go",
    markersize=10,
    zorder=11,
    label="Parent Start",
)
plt.plot(
    t_coords_classical[-1],
    parent_x_classical_mean[-1],
    "gs",
    markersize=10,
    zorder=11,
    label="Parent End",
)
plt.plot(
    t_coords_classical[0],
    child_x_classical[0],
    "o",
    color="orange",
    markersize=10,
    zorder=11,
    label="Child Start",
)
plt.plot(
    t_coords_classical[-1],
    child_x_classical[-1],
    "s",
    color="yellow",
    markersize=10,
    zorder=11,
    label="Child End",
)


# 4. Light cones for context (from parent's start)
cone_t_parent_start = np.linspace(t_start, t_end, 100)
plt.plot(
    cone_t_parent_start,
    parent_x_classical_mean[0] + (cone_t_parent_start - t_start),
    "k--",
    alpha=0.4,
    lw=1,
    label="Light Cone Boundary (Parent Start)",
)
plt.plot(
    cone_t_parent_start,
    parent_x_classical_mean[0] - (cone_t_parent_start - t_start),
    "k--",
    alpha=0.4,
    lw=1,
)

plt.xlabel("Semantic Time (t)")
plt.ylabel("Semantic Spatial Coordinate (x)")
plt.title(
    f"Path Integral Analogy with Semantic Gravity & Spin\n(Gs={G_s}, Child mass={m_c})"
)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="lower right")

plt.grid(True)
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)

# Adjust limits to see the orbit well
min_x_vals = np.concatenate((
    parent_x_classical_mean,
    child_x_classical,
))
max_x_vals = np.concatenate((
    parent_x_classical_mean,
    child_x_classical,
))
plt.ylim(np.min(min_x_vals) - 0.2, np.max(max_x_vals) + 0.2)


plt.figtext(
    0.5,
    0.005,
    "Red/Magenta: Classical paths under 'semantic gravity'. Child 'orbits' (oscillates around) parent.\n"
    "Blue/Cyan: Sample fluctuating paths around these new classical trajectories.\n"
    "Prediction is 'easier' as fluctuations are now around a more defined (though complex) dynamic.",
    ha="center",
    va="bottom",
    fontsize=9,
    bbox={"facecolor": "lightgray", "alpha": 0.7, "pad": 5},
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
# %% ======================================================
# %%[markdown]
# --- Semantic Gravity Simulation ---
#
# **Interpretation:**
# - The Parent (red) follows a pre-defined mean path (slight drift in x).
# - The Child (magenta) is attracted to the Parent by 'semantic gravity', leading to an oscillatory 'orbital' motion around the Parent's path.
# - This orbital/oscillatory path is the *new classical path* for the Child.
# - The blue/cyan fluctuating paths show deviations (perturbations) around these new classical paths.
#
# **Predictability:**
# - If this gravitational dynamic is the true underlying 'physics' of the semantic link, then predicting the Child's state given the Parent's involves understanding this orbit.
# - The 'information leak' (system settling into gravitational dominance) has replaced broad random fluctuations with a more structured, albeit dynamic, behavior.
# - The fluctuations (perturbation strength) are now relative to this more complex, but deterministic, classical orbit, potentially making the *relative uncertainty* smaller compared to the overall dynamic range of the orbit itself.

# %% ======================================================
# [link!](https://claude.ai/public/artifacts/6c6b533b-e0f0-4be8-bbee-1aa8aa5b1351)


# %% ======================================================
# Shifted Negative Sigmoid
def f(x, k):
    """Defines the function f(x) = -1/(1 + e^(-kx)) + 0.5."""
    return -1 / (1 + np.exp(-k * x)) + 0.5


# Define parameters for the plot
x_vals = np.linspace(-10, 10, 200)  # x range
k_values = [
    0.5,
    1.0,
    2.0,
    5.0,
]  # Different k values to show steepness

plt.figure(figsize=(8, 6))

for k in k_values:
    y_vals = f(x_vals, k)
    plt.plot(x_vals, y_vals, label=f"k = {k}")

plt.title("Plot of f(x) = -1/(1 + e^(-kx)) + 0.5 for Different k")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color="gray", lw=0.5)  # Add y=0 line
plt.axvline(0, color="gray", lw=0.5)  # Add x=0 line
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

#%%[markdown]
# --- Analysis of the Function f(x) ---
#
# The function f(x) = -1 / (1 + e^(-kx)) + 0.5 is a scaled and shifted sigmoid function.
# It is antisymmetric around (0, 0), meaning f(-x) = -f(x).
#
# ## Key properties:
# - As x approaches infinity, e^(-kx) approaches 0 (assuming k > 0), so f(x) approaches -1/(1+0) + 0.5 = -1 + 0.5 = -0.5
# - As x approaches negative infinity, e^(-kx) approaches infinity (assuming k > 0), so f(x) approaches -1/(large number) + 0.5 = 0 + 0.5 = 0.5
# - At x = 0, f(0) = -1/(1 + e^0) + 0.5 = -1/(1+1) + 0.5 = -0.5 + 0.5 = 0
#
# ## Parameter 'k' Behavior
# The parameter 'k' controls the steepness of the curve around x=0:
# - Larger k: The curve is steeper (transition from 0.5 to -0.5 is more rapid)
# - Smaller k: The curve is less steep (transition is more gradual)
#
# ## Applications
# This type of function can be used in various contexts, such as:
# - As an activation function (like tanh, which it resembles when scaled/shifted) in neural networks
# - Modeling probabilities or transitions that saturate at limits
# - Creating a smooth, S-shaped transition between two values (-0.5 and 0.5 in this case)

# %% ======================================================
# --- Simulation Parameters ---
# Common
t_start = 0.5
t_end = 2.0  # Extended time to see more evolution
num_t_steps = 300
dt_sim = (t_end - t_start) / num_t_steps

m_p = 1.0
m_c = 0.1
epsilon_soft = 0.02

parent_x_start = -0.2
parent_vx_mean = 0.05  # Parent's gentle drift

child_x_initial_offset = 0.1
child_vx_initial_relative = (
    -0.1
)  # Initial relative velocity of child to parent

num_path_segments_fluctuations = num_t_steps
num_sample_paths = 15
perturbation_strength_parent = 0.015
perturbation_strength_child = 0.02

# Rule-specific parameters
# Rule 1: Gravity
G_s_gravity = 0.01

# Rule 2: Repulsion
G_s_repulsion = (
    0.005  # Repulsion constant (can be different from gravity)
)

# Rule 3: Guided Channel/Wave
A_wave = 0.15  # Amplitude of the wave relative to parent
omega_wave = 4.0  # Angular frequency of the wave
phi_wave = np.pi / 2  # Initial phase of the wave
k_channel = 5.0  # Spring constant pulling child to the wave track
damping_channel = 0.5  # Damping for stability in following the wave


# --- Helper function for Brownian Bridge (from previous examples) ---
def generate_brownian_bridge_component(num_steps, scale):
    if num_steps == 0:
        return np.array([0.0])
    dt_norm = 1.0 / num_steps
    dW_scaled = (
        scale * np.sqrt(dt_norm) * np.random.normal(0, 1, num_steps)
    )
    W_scaled = np.zeros(num_steps + 1)
    W_scaled[1:] = np.cumsum(dW_scaled)
    s_coords = np.linspace(0, 1, num_steps + 1)
    return W_scaled - s_coords * W_scaled[-1]


# --- Function to simulate child's classical path for a given rule ---
def simulate_child_classical_path(
    rule_type, t_coords, parent_x_mean, parent_vx_mean_val
):
    child_x_classical = np.zeros_like(t_coords)

    # Initial child velocity (absolute)
    # child_vx_abs_current = parent_vx_mean_val + child_vx_initial_relative
    child_vx_abs_current = (
        parent_vx_mean_val + child_vx_initial_relative
    )

    child_x_classical[0] = parent_x_mean[0] + child_x_initial_offset

    for i in range(num_t_steps):
        xp_i = parent_x_mean[i]
        xc_i = child_x_classical[i]
        t_i = t_coords[i]

        accel_c_x = 0.0

        if rule_type == "gravity":
            r_x = xc_i - xp_i
            dist_cubed_soft = (r_x**2 + epsilon_soft**2) ** (1.5)
            if abs(dist_cubed_soft) > 1e-9:
                force_g_x = (
                    -G_s_gravity * m_p * m_c * r_x / dist_cubed_soft
                )
                accel_c_x = force_g_x / m_c

        elif rule_type == "repulsion":
            r_x = xc_i - xp_i
            dist_cubed_soft = (r_x**2 + epsilon_soft**2) ** (1.5)
            if abs(dist_cubed_soft) > 1e-9:
                force_r_x = (
                    +G_s_repulsion * m_p * m_c * r_x / dist_cubed_soft
                )  # Note the '+'
                accel_c_x = force_r_x / m_c

        elif rule_type == "wave_channel":
            target_x_offset = A_wave * np.sin(
                omega_wave * (t_i - t_start) + phi_wave
            )
            x_child_target = xp_i + target_x_offset

            # Velocity of the target wave path (analytical derivative)
            # vx_parent is parent_vx_mean_val
            # d/dt (target_x_offset) = A_wave * omega_wave * cos(...)
            vx_target_offset_derivative = (
                A_wave
                * omega_wave
                * np.cos(omega_wave * (t_i - t_start) + phi_wave)
            )
            vx_child_target_wave = (
                parent_vx_mean_val + vx_target_offset_derivative
            )

            force_channel_x = -k_channel * (
                xc_i - x_child_target
            ) - damping_channel * (
                child_vx_abs_current - vx_child_target_wave
            )
            accel_c_x = force_channel_x / m_c

        elif (
            rule_type == "no_interaction"
        ):  # Child drifts with initial relative velocity
            # In this frame, acceleration is zero if we consider child_vx_abs_current
            # to be already set with its initial impulse relative to parent's drift.
            # Or, more simply, its acceleration relative to an inertial frame is 0.
            # No additional acceleration due to interaction.
            pass

        child_vx_abs_current += accel_c_x * dt_sim
        child_x_classical[i + 1] = (
            child_x_classical[i] + child_vx_abs_current * dt_sim
        )

    return child_x_classical


# --- Parent's Mean Classical Trajectory (same for all scenarios) ---
t_coords_classical = np.linspace(t_start, t_end, num_t_steps + 1)
parent_x_classical_mean = np.zeros_like(t_coords_classical)
parent_x_classical_mean[0] = parent_x_start
for i in range(num_t_steps):
    parent_x_classical_mean[i + 1] = (
        parent_x_classical_mean[i] + parent_vx_mean * dt_sim
    )

# --- Setup Scenarios and Plot ---
scenarios = [
    {"rule": "gravity", "title": "Rule: Gravitational Attraction"},
    {"rule": "repulsion", "title": "Rule: Repulsive Force"},
    {"rule": "wave_channel", "title": "Rule: Guided Wave Channel"},
    {
        "rule": "no_interaction",
        "title": "Rule: No Interaction (Inertial Drift)",
    },
]

fig, axes = plt.subplots(
    2, 2, figsize=(16, 14), sharex=True
)  # Removed sharey for better individual scaling
axes = axes.ravel()

for ax_idx, scenario_info in enumerate(scenarios):
    ax = axes[ax_idx]
    rule = scenario_info["rule"]
    title = scenario_info["title"]

    child_x_classical = simulate_child_classical_path(
        rule,
        t_coords_classical,
        parent_x_classical_mean,
        parent_vx_mean,
    )

    # Plot Classical Paths
    ax.plot(
        t_coords_classical,
        parent_x_classical_mean,
        "r-",
        lw=2,
        label="Parent Mean Classical Path",
        zorder=10,
    )
    ax.plot(
        t_coords_classical,
        child_x_classical,
        "m-",
        lw=2,
        label="Child Rule-Defined Classical Path",
        zorder=10,
    )

    # Plot Fluctuating Paths
    for i in range(num_sample_paths):
        parent_fluctuations = generate_brownian_bridge_component(
            num_t_steps, perturbation_strength_parent
        )
        parent_x_fluctuating = (
            parent_x_classical_mean + parent_fluctuations
        )

        child_fluctuations = generate_brownian_bridge_component(
            num_t_steps, perturbation_strength_child
        )
        child_x_fluctuating = child_x_classical + child_fluctuations

        ax.plot(
            t_coords_classical,
            parent_x_fluctuating,
            "b-",
            lw=0.5,
            alpha=0.25,
        )
        ax.plot(
            t_coords_classical,
            child_x_fluctuating,
            "c-",
            lw=0.5,
            alpha=0.25,
        )

    # Dummy plots for unified legend items for fluctuations
    if ax_idx == 0:
        ax.plot(
            [],
            [],
            "b-",
            lw=0.5,
            alpha=0.6,
            label="Parent Fluctuating Sample",
        )
        ax.plot(
            [],
            [],
            "c-",
            lw=0.5,
            alpha=0.6,
            label="Child Fluctuating Sample",
        )

    # Light cones from parent's start (same for all)
    cone_t_parent_start = np.linspace(t_start, t_end, 100)
    ax.plot(
        cone_t_parent_start,
        parent_x_classical_mean[0] + (cone_t_parent_start - t_start),
        "k--",
        alpha=0.3,
        lw=1,
        label="Light Cone (Parent Start)" if ax_idx == 0 else None,
    )
    ax.plot(
        cone_t_parent_start,
        parent_x_classical_mean[0] - (cone_t_parent_start - t_start),
        "k--",
        alpha=0.3,
        lw=1,
    )

    ax.set_xlabel("Semantic Time (t)")
    ax.set_ylabel("Semantic Spatial Coordinate (x)")
    ax.set_title(title)
    ax.grid(True)

    # Determine y-limits dynamically for each subplot
    all_y_paths = np.concatenate((
        parent_x_classical_mean,
        child_x_classical,
    ))
    min_y, max_y = np.min(all_y_paths), np.max(all_y_paths)
    padding_y = 0.1 * (max_y - min_y) if (max_y - min_y) > 0 else 0.1
    ax.set_ylim(
        min_y - padding_y - 0.1, max_y + padding_y + 0.1
    )  # Extra padding for fluctuations


fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
fig.suptitle(
    "Path Integral Analogy: Child's Dynamics under Different Semantic 'Rules'",
    fontsize=16,
)
plt.tight_layout(
    rect=[0, 0.06, 1, 0.95]
)  # Adjust for suptitle and legend
plt.show()


# --- Simulation Parameters (from previous gravity example) ---
# Common
t_start = 0.5
t_end = (
    1.5  # Adjusted from 2.0 to focus on the first "swoop" for tanh
)
num_t_steps = 200  # Increased for smoother classical path
dt_sim = (t_end - t_start) / num_t_steps

m_p = 1.0
m_c = 0.1
epsilon_soft = 0.02
G_s_gravity = 0.01

parent_x_start = -0.2
parent_vx_mean = 0.05

child_x_initial_offset = 0.1
child_vx_initial_relative = (
    -0.2
)  # This value is key for the "swoop" shape

# --- Parent's Mean Classical Trajectory ---
t_coords_classical = np.linspace(t_start, t_end, num_t_steps + 1)
parent_x_classical_mean = np.zeros_like(t_coords_classical)
parent_x_classical_mean[0] = parent_x_start
for i in range(num_t_steps):
    parent_x_classical_mean[i + 1] = (
        parent_x_classical_mean[i] + parent_vx_mean * dt_sim
    )


# --- Simulate Child's Classical Path under Gravity (from previous script) ---
def simulate_child_classical_path_gravity(
    t_coords, parent_x_mean_path, parent_vx_mean_val
):
    child_x_classical_sim = np.zeros_like(t_coords)
    child_vx_abs_current = (
        parent_vx_mean_val + child_vx_initial_relative
    )
    child_x_classical_sim[0] = (
        parent_x_mean_path[0] + child_x_initial_offset
    )

    for i in range(num_t_steps):
        xp_i = parent_x_mean_path[i]
        xc_i = child_x_classical_sim[i]

        r_x = xc_i - xp_i
        dist_cubed_soft = (r_x**2 + epsilon_soft**2) ** (1.5)
        accel_c_x = 0.0
        if abs(dist_cubed_soft) > 1e-9:
            force_g_x = (
                -G_s_gravity * m_p * m_c * r_x / dist_cubed_soft
            )
            accel_c_x = force_g_x / m_c

        child_vx_abs_current += accel_c_x * dt_sim
        child_x_classical_sim[i + 1] = (
            child_x_classical_sim[i] + child_vx_abs_current * dt_sim
        )
    return child_x_classical_sim


child_x_gravitational_path = simulate_child_classical_path_gravity(
    t_coords_classical, parent_x_classical_mean, parent_vx_mean
)


# --- Define Model Functions for Fitting ---
def model_tanh(t, A, B, t0, C):
    """A * tanh(B * (t - t0)) + C"""
    return A * np.tanh(B * (t - t0)) + C


def model_bell_curve(t, Amp, mu, sigma, offset):
    """Gaussian-like: Amp * exp( -(t - mu)^2 / (2 * sigma^2) ) + offset"""
    return Amp * np.exp(-((t - mu) ** 2) / (2 * sigma**2)) + offset


def model_sigmoid_derivative_like(t, Amp, k, t0, offset):
    """
    Shape similar to derivative of sigmoid: Amp * k * exp(-k*(t-t0)) / (1 + exp(-k*(t-t0)))**2 + offset
    This can be tricky to fit directly. Let's use the Gaussian (model_bell_curve)
    as it's more standard for bell shapes.
    If we want to try this:
    val = k * (t - t0)
    return Amp * k * np.exp(-val) / (1 + np.exp(-val))**2 + offset
    For our purpose, model_bell_curve is a good proxy for "bell-shaped derivative of sigmoid".
    """
    # Using Gaussian as a proxy for bell shape
    return model_bell_curve(
        t, Amp, k, t0, offset
    )  # k here is mu, t0 is sigma for bell


# --- Perform Curve Fitting ---

# Fit Tanh Model (expecting an "upside-down" or negative A)
# Initial guesses (p0) are important for good fits!
# A: Amplitude (negative for upside down)
# B: Steepness (positive)
# t0: Center of transition (around midpoint of t_coords)
# C: Vertical offset (around mean of x values)
initial_guess_tanh = [
    -0.1,
    5,
    np.mean(t_coords_classical),
    np.mean(child_x_gravitational_path),
]
try:
    params_tanh, covariance_tanh = curve_fit(
        model_tanh,
        t_coords_classical,
        child_x_gravitational_path,
        p0=initial_guess_tanh,
        maxfev=5000,
    )
    y_fit_tanh = model_tanh(t_coords_classical, *params_tanh)
    fit_success_tanh = True
    print("Tanh fit parameters (A, B, t0, C):", params_tanh)
except RuntimeError:
    print("Tanh fit failed to converge.")
    y_fit_tanh = np.full_like(
        child_x_gravitational_path, np.nan
    )  # Placeholder if fit fails
    fit_success_tanh = False


# Fit Bell Curve Model (expecting a negative Amp for the "dip" part of the oscillation)
# Amp: Amplitude of peak/dip (negative for dip)
# mu: Center of peak/dip (time)
# sigma: Width of peak/dip
# offset: Vertical offset
# We might be fitting the first "dip" of the oscillation
first_dip_approx_time = t_coords_classical[
    np.argmin(child_x_gravitational_path)
]  # Approx time of min
initial_guess_bell = [
    np.min(child_x_gravitational_path)
    - np.mean(child_x_gravitational_path),  # Amplitude (negative)
    first_dip_approx_time,  # mu
    0.2,  # sigma (guess for width)
    np.mean(child_x_gravitational_path),  # offset
]
try:
    params_bell, covariance_bell = curve_fit(
        model_bell_curve,
        t_coords_classical,
        child_x_gravitational_path,
        p0=initial_guess_bell,
        maxfev=5000,
    )
    y_fit_bell = model_bell_curve(t_coords_classical, *params_bell)
    fit_success_bell = True
    print(
        "Bell curve fit parameters (Amp, mu, sigma, offset):",
        params_bell,
    )
except RuntimeError:
    print("Bell curve fit failed to converge.")
    y_fit_bell = np.full_like(
        child_x_gravitational_path, np.nan
    )  # Placeholder
    fit_success_bell = False

# --- Visualization ---
plt.figure(figsize=(12, 7))
plt.plot(
    t_coords_classical,
    parent_x_classical_mean,
    "r-",
    lw=1.5,
    alpha=0.7,
    label="Parent Mean Classical Path",
)
plt.plot(
    t_coords_classical,
    child_x_gravitational_path,
    "m-",
    lw=2.5,
    label="Child Gravitational Path (Simulated)",
)

if fit_success_tanh:
    plt.plot(
        t_coords_classical,
        y_fit_tanh,
        "b--",
        lw=2,
        label=f"Fitted Tanh Model (A={params_tanh[0]:.2f})",
    )
if fit_success_bell:
    plt.plot(
        t_coords_classical,
        y_fit_bell,
        "g:",
        lw=2,
        label=f"Fitted Bell Curve Model (Amp={params_bell[0]:.2f})",
    )


plt.xlabel("Semantic Time (t)")
plt.ylabel("Semantic Spatial Coordinate (x)")
plt.title("Fitting Intuitive Functions to Child's Gravitational Path")
plt.legend()
plt.grid(True)
plt.show()

# --- Discussion of Results ---
print("\n--- Discussion ---")
if fit_success_tanh:
    print(
        f"Tanh Model: The parameter A={params_tanh[0]:.3f} suggests the overall direction of the transition."
    )
    print(
        "  If A is negative, it's an 'upside-down' tanh, matching the initial downward swoop."
    )
    print(
        "  How well does it capture the whole oscillation? Likely only the first major 'bend'."
    )
else:
    print(
        "Tanh Model: Fit was not successful. The shape might be too complex for a single tanh."
    )

if fit_success_bell:
    print(
        f"Bell Curve Model: Amplitude={params_bell[0]:.3f}, Center (mu)={params_bell[1]:.3f} (time)."
    )
    print(
        "  If Amp is negative, it's fitting a 'dip' or 'valley' in the path."
    )
    print(
        "  This model is good for single peaks/valleys, so it might capture one part of the oscillation well."
    )
else:
    print(
        "Bell Curve Model: Fit was not successful. The shape might be too complex for a single bell curve over the whole range."
    )

print("\nGeneral Observations:")
print(
    "- A single tanh or bell curve is unlikely to perfectly model a full oscillatory gravitational path."
)
print(
    "- However, they might capture *segments* of the path quite well, confirming the visual intuition for those parts."
)
print(
    "- For the tanh, the fit focuses on the primary transition. For the bell curve, it focuses on a peak or valley."
)
print(
    "- To model the full oscillation, one would typically use sums of sines/cosines (Fourier series) or solutions to the actual differential equations of motion."
)

# --- Assume these functions are defined as before ---
# model_tanh(t, A, B, t0, C)
# simulate_child_classical_path_gravity_for_pair(parent_info, child_info, Gs, t_coords)
#    -> This new function would need to figure out masses, initial conditions, etc.
#       based on 'parent_info' and 'child_info' (e.g., their word embeddings)


def predict_is_child_by_tanh_swoop(
    parent_concept_data,
    child_concept_data,
    simulation_time_coords,
    tanh_fit_initial_guess,
    A_threshold=-0.05,
):  # Example threshold
    """
    Predicts if child is a child of parent based on the 'A' parameter
    of a fitted tanh curve to their simulated gravitational interaction.
    """

    # 1. Simulate the classical path of the child under "gravity" from the parent
    # This is the most complex, hypothetical step:
    # It needs to derive m_p, m_c, initial_offset, initial_relative_velocity, Gs
    # from parent_concept_data and child_concept_data.
    # For now, let's assume we can get a trajectory:
    # child_x_simulated = simulate_child_classical_path_gravity_for_pair(
    #     parent_concept_data, child_concept_data, Gs_universal, simulation_time_coords
    # )

    # For demonstration, let's reuse our fixed simulation from before
    # In a real system, this would be dynamic based on the input pair
    t_coords_classical_demo = np.linspace(
        0.5, 1.5, 201
    )  # Match previous example
    parent_x_classical_mean_demo = -0.2 + 0.05 * (
        t_coords_classical_demo - 0.5
    )  # Simplified parent path

    # --- Simulate Child's Classical Path under Gravity (from previous script) ---
    # Reusing the exact simulation logic from the image for consistency in this thought experiment
    m_p_demo = 1.0
    m_c_demo = 0.1
    epsilon_soft_demo = 0.02
    G_s_gravity_demo = 0.01
    child_x_initial_offset_demo = 0.1
    child_vx_initial_relative_demo = -0.2

    child_x_simulated = np.zeros_like(t_coords_classical_demo)
    child_vx_abs_current = (
        parent_x_classical_mean_demo[1]
        - parent_x_classical_mean_demo[0]
    ) / (
        t_coords_classical_demo[1] - t_coords_classical_demo[0]
    ) + child_vx_initial_relative_demo
    child_x_simulated[0] = (
        parent_x_classical_mean_demo[0] + child_x_initial_offset_demo
    )
    dt_sim_demo = (
        t_coords_classical_demo[1] - t_coords_classical_demo[0]
    )

    for i in range(len(t_coords_classical_demo) - 1):
        xp_i = parent_x_classical_mean_demo[i]
        xc_i = child_x_simulated[i]
        r_x = xc_i - xp_i
        dist_cubed_soft = (r_x**2 + epsilon_soft_demo**2) ** (1.5)
        accel_c_x = 0.0
        if abs(dist_cubed_soft) > 1e-9:
            force_g_x = (
                -G_s_gravity_demo
                * m_p_demo
                * m_c_demo
                * r_x
                / dist_cubed_soft
            )
            accel_c_x = force_g_x / m_c_demo
        child_vx_abs_current += accel_c_x * dt_sim_demo
        child_x_simulated[i + 1] = (
            child_x_simulated[i] + child_vx_abs_current * dt_sim_demo
        )
    # End of reused simulation logic

    # 2. Fit the Tanh Model
    try:
        # Use more robust initial guesses if possible, or derive them from data
        default_initial_guess_tanh = [
            (np.min(child_x_simulated) - np.max(child_x_simulated))
            / 2,  # A
            5,  # B
            np.mean(t_coords_classical_demo),  # t0
            np.mean(child_x_simulated),  # C
        ]
        params_tanh, _ = curve_fit(
            model_tanh,
            t_coords_classical_demo,
            child_x_simulated,
            p0=tanh_fit_initial_guess
            if tanh_fit_initial_guess
            else default_initial_guess_tanh,
            maxfev=5000,
        )
        A_fitted = params_tanh[0]
        fit_successful = True
    except RuntimeError:
        print(f"Tanh fit failed for this pair.")
        A_fitted = 0  # Or some other default indicating failure/no clear pattern
        fit_successful = False

    # 3. Classify based on A_fitted
    if fit_successful and A_fitted < A_threshold:
        # Optional: could also check B for reasonable steepness,
        # and if t0 is within the simulation time range.
        return "Yes, likely child (strong attractive swoop)"
    elif fit_successful and A_fitted > abs(
        A_threshold
    ):  # Example for repulsion
        return "No, likely repulsive"
    else:
        return "Uncertain / No clear child-like attractive swoop"


# --- Example Usage (Conceptual) ---
# Define a 'model_tanh' function as before
def model_tanh(t, A, B, t0, C):
    return A * np.tanh(B * (t - t0)) + C


# This is where we'd get real data for parent/child if we had an embedding space
# For now, we'll just use the parameters that generated the image
parent_data_example = {
    "name": "mammal",
    "embedding_vec": [0.1, 0.2],
    "mass_proxy": 1.0,
}
child_data_example_dog = {
    "name": "dog",
    "embedding_vec": [0.15, 0.15],
    "mass_proxy": 0.1,
}
child_data_example_rock = {
    "name": "rock",
    "embedding_vec": [0.8, 0.9],
    "mass_proxy": 0.5,
}  # Unrelated

# Use the initial guesses from our successful fit if possible
# params_tanh from image: A=-0.09, B (derived), t0 (derived), C (derived)
# From your image's legend, A = -0.09. We don't have B, t0, C directly from the legend,
# but the code that generated it had `initial_guess_tanh = [-0.1, 5, np.mean(t_coords_classical), np.mean(child_x_gravitational_path)]`
# And the fitted params were (A, B, t0, C): [-0.08961384  6.19520546  0.95711156 -0.19078084] for a similar run.
# Let's use these as a good starting point for a "known good" case.
example_tanh_fit_initial_guess = [-0.09, 6.0, 0.95, -0.19]
example_simulation_time_coords = np.linspace(
    0.5, 1.5, 201
)  # From image-generating code

prediction_dog = predict_is_child_by_tanh_swoop(
    parent_data_example,
    child_data_example_dog,
    example_simulation_time_coords,
    example_tanh_fit_initial_guess,
    A_threshold=-0.05,  # If fitted A is more negative than this
)
print(f"Prediction for (mammal, dog): {prediction_dog}")

# To test (mammal, rock), we'd need a way to simulate its path.
# If 'rock' had no gravitational attraction or was repulsive, A_fitted would be different.
# For now, we can't run that without defining how `simulate_child_classical_path_gravity_for_pair`
# would handle very different concepts.
# If we assume "rock" results in A_fitted = 0.02 (e.g. slight repulsion or drift):
# This is a conceptual placeholder:
prediction_rock = "Uncertain / No clear child-like attractive swoop"  # if A_fitted was positive
print(f"Prediction for (mammal, rock): {prediction_rock}")

# --- Configuration & Global Parameters ---
SIMULATION_T_START = 0.0
SIMULATION_T_END = 1.0  # Shorter time to focus on initial swoop
SIMULATION_NUM_T_STEPS = 100
DT_SIM = (
    SIMULATION_T_END - SIMULATION_T_START
) / SIMULATION_NUM_T_STEPS

# Semantic Physics Parameters (THESE ARE HYPOTHETICAL AND NEED TUNING/DERIVATION)
G_S_GRAVITY = (
    0.05  # Universal semantic gravitational constant (example)
)
DEFAULT_M_P = 1.0  # Default parent mass
DEFAULT_M_C = 0.1  # Default child mass (lighter to be more affected)
EPSILON_SOFT = 0.01  # Softening for gravity

# Embedding model (using a small pre-trained one for demonstration if available)
# Or use a placeholder if no model is loaded
EMBEDDING_DIM = (
    50  # Example dimension if using simple random embeddings
)

USE_GLOVE = False  # Set to True if GloVe embeddings are available
# --- 1. Data Preparation ---


def get_synset_embedding(synset):
    """Gets a vector embedding for a synset."""
    if USE_GLOVE:
        # Average embeddings of lemmas in the synset
        vecs = []
        for lemma in synset.lemmas():
            try:
                vecs.append(
                    glove_vectors[
                        lemma.name().lower().replace("_", " ")
                    ]
                )  # Try space, then underscore
            except KeyError:
                try:
                    vecs.append(glove_vectors[lemma.name().lower()])
                except KeyError:
                    pass  # Lemma not in GloVe vocab
        if vecs:
            return np.mean(vecs, axis=0)
        else:  # Fallback if no lemmas are in vocab
            # print(f"Warning: No GloVe vectors for lemmas in {synset.name()}. Using random.")
            return (
                np.random.rand(EMBEDDING_DIM) * 0.1
            )  # Small random vector
    else:  # Fallback to random embeddings
        # In a real scenario, you'd want consistent random embeddings for the same synset
        # This simple version will generate new random ones each time, which is bad for training.
        # For a real use case with random, pre-generate and store them.
        return np.random.rand(EMBEDDING_DIM) * 0.1


def get_wordnet_pairs(num_positive=1000, num_negative_ratio=1):
    """Gets positive (hypernym-hyponym) and negative pairs from WordNet."""
    all_synsets = list(
        wn.all_synsets("n")
    )  # Focus on nouns for simplicity
    if not all_synsets:
        print(
            "Error: NLTK WordNet data might not be downloaded. Run nltk.download('wordnet')"
        )
        return [], []

    random.shuffle(all_synsets)

    positive_pairs = []
    processed_parents = set()

    for s_parent in all_synsets:
        if s_parent in processed_parents:
            continue
        hyponyms = s_parent.hyponyms()
        if hyponyms:
            for s_child in hyponyms:
                positive_pairs.append((s_parent, s_child))
                if len(positive_pairs) >= num_positive:
                    break
            processed_parents.add(
                s_parent
            )  # Avoid over-representing prolific parents
        if len(positive_pairs) >= num_positive:
            break

    num_negative = int(len(positive_pairs) * num_negative_ratio)
    negative_pairs = []
    attempts = 0
    max_attempts = num_negative * 100  # Safety break

    while (
        len(negative_pairs) < num_negative and attempts < max_attempts
    ):
        s1, s2 = random.sample(all_synsets, 2)
        attempts += 1
        if s1 == s2:
            continue
        # Check if s2 is NOT a hyponym of s1 AND s1 is NOT a hyponym of s2
        # Also ensure they are not the same path to root (e.g. great-grandparent)
        # A simple check: are they direct parent/child?
        s1_hyponyms = set(s1.hyponyms())
        s2_hyponyms = set(s2.hyponyms())
        if s2 not in s1_hyponyms and s1 not in s2_hyponyms:
            # A more robust check would trace paths, but this is a start
            common_hypernyms = set(s1.common_hypernyms(s2))
            if (
                not common_hypernyms or len(common_hypernyms) < 2
            ):  # Avoid too closely related things by chance
                negative_pairs.append((
                    s1,
                    s2,
                ))  # Order matters if we use it for parent/child roles

    print(
        f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs."
    )
    return positive_pairs, negative_pairs


# --- 2. Semantic Physics Simulation ---


def derive_initial_conditions(
    parent_synset, child_synset, parent_emb, child_emb
):
    """
    HYPOTHETICAL: Derives simulation parameters from synsets and embeddings.
    This is where most of the "semantic physics" assumptions live.
    """
    # Masses: could be based on depth, number of children/parents, or fixed
    m_p = DEFAULT_M_P
    # m_c could be inversely proportional to its "generality" or fixed
    m_c = DEFAULT_M_C * (
        1 + random.uniform(-0.05, 0.05)
    )  # Add small noise

    # Initial spatial offset: based on embedding distance?
    # For 1D simulation, we need to project. Let's use a component or distance.
    if parent_emb is not None and child_emb is not None:
        # Cosine distance is 1 - cosine_similarity. Ranges [0, 2].
        # distance = np.dot(parent_emb, child_emb) / (np.linalg.norm(parent_emb) * np.linalg.norm(child_emb))
        # initial_offset = 0.1 + 0.2 * (1 - distance) # Scale cosine similarity to an offset
        # Let's use Euclidean distance for simplicity
        initial_offset = np.linalg.norm(parent_emb - child_emb) / (
            np.sqrt(EMBEDDING_DIM) * 2
        )  # Normalize roughly
        initial_offset = np.clip(
            initial_offset * 0.5, 0.05, 0.3
        )  # Ensure some offset, but not too large
    else:
        initial_offset = 0.1 + random.uniform(-0.05, 0.05)

    # Initial relative velocity:
    # Negative to induce a "swoop" towards the parent. Magnitude could vary.
    # Perhaps if they are "far" (large initial_offset), give a larger initial pull.
    initial_vx_rel = (
        -0.15 - 0.1 * initial_offset + random.uniform(-0.05, 0.05)
    )

    return m_p, m_c, initial_offset, initial_vx_rel


def simulate_child_trajectory_for_pair(parent_synset, child_synset):
    """Simulates child's path given a parent-child pair."""
    parent_emb = get_synset_embedding(parent_synset)
    child_emb = get_synset_embedding(child_synset)

    m_p, m_c, initial_offset, initial_vx_rel = (
        derive_initial_conditions(
            parent_synset, child_synset, parent_emb, child_emb
        )
    )

    t_coords = np.linspace(
        SIMULATION_T_START,
        SIMULATION_T_END,
        SIMULATION_NUM_T_STEPS + 1,
    )

    # Parent's mean classical path (fixed at x=0 for simplicity in this 1D relative sim)
    parent_x_mean = np.zeros_like(t_coords)
    parent_vx_mean_val = 0.0  # Parent is static in x

    # Child's classical trajectory
    child_x_classical = np.zeros_like(t_coords)
    child_vx_abs_current = (
        parent_vx_mean_val + initial_vx_rel
    )  # Absolute velocity
    child_x_classical[0] = parent_x_mean[0] + initial_offset

    for i in range(SIMULATION_NUM_T_STEPS):
        xp_i = parent_x_mean[i]
        xc_i = child_x_classical[i]

        r_x = xc_i - xp_i  # Child's position relative to parent
        dist_cubed_soft = (r_x**2 + EPSILON_SOFT**2) ** (1.5)
        accel_c_x = 0.0
        if abs(dist_cubed_soft) > 1e-12:  # Avoid division by zero
            force_g_x = (
                -G_S_GRAVITY * m_p * m_c * r_x / dist_cubed_soft
            )
            accel_c_x = force_g_x / m_c

        child_vx_abs_current += accel_c_x * DT_SIM
        child_x_classical[i + 1] = (
            child_x_classical[i] + child_vx_abs_current * DT_SIM
        )

    return t_coords, child_x_classical


# --- 3. Feature Extraction ---


def model_tanh(t, A, B, t0, C):
    """A * tanh(B * (t - t0)) + C"""
    return A * np.tanh(B * (t - t0)) + C


def extract_features_from_trajectory(t_coords, child_x_trajectory):
    """Fits tanh and extracts parameter A."""
    # Initial guesses are crucial
    mean_x = np.mean(child_x_trajectory)
    delta_x = np.max(child_x_trajectory) - np.min(child_x_trajectory)
    if delta_x < 1e-6:
        delta_x = 0.1  # Avoid zero delta_x

    # Attempt to guess direction for A
    initial_A_guess = -delta_x / 2  # Guessing a downward swoop
    if child_x_trajectory[-1] > child_x_trajectory[0] and abs(
        child_x_trajectory[0] - mean_x
    ) > abs(child_x_trajectory[-1] - mean_x):
        initial_A_guess = (
            delta_x / 2
        )  # upward swoop if it ends higher and starts lower rel to mean

    initial_guess = [
        initial_A_guess,  # A
        5
        / (t_coords[-1] - t_coords[0]),  # B (scales with time range)
        np.mean(t_coords),  # t0
        mean_x,  # C
    ]

    try:
        params, covariance = curve_fit(
            model_tanh,
            t_coords,
            child_x_trajectory,
            p0=initial_guess,
            maxfev=10000,
        )
        A_fitted = params[0]
        # Could also calculate goodness of fit (e.g., MSE or R^2) as a feature
        # y_pred = model_tanh(t_coords, *params)
        # mse = np.mean((child_x_trajectory - y_pred)**2)
        return [A_fitted]  # Return as a list of features
    except RuntimeError:
        # print("Tanh fit failed for a trajectory.")
        return [
            0.0
        ]  # Default feature if fit fails (e.g., A=0 means no clear swoop)


# --- 4. Train Classifier ---
# (This will be done after generating all features)

# --- Main Execution ---
if __name__ == "__main__":
    print("Preparing WordNet pairs...")
    # Using small numbers for quick test, increase for real experiment
    positive_pairs, negative_pairs = get_wordnet_pairs(
        num_positive=200, num_negative_ratio=1.5
    )

    if not positive_pairs or not negative_pairs:
        print("Not enough pairs generated. Exiting.")
        exit()

    dataset = []
    labels = []

    print("Simulating and extracting features for positive pairs...")
    for i, (p_syn, c_syn) in enumerate(positive_pairs):
        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1}/{len(positive_pairs)} positive pairs..."
            )
        t_coords, child_x = simulate_child_trajectory_for_pair(
            p_syn, c_syn
        )
        features = extract_features_from_trajectory(t_coords, child_x)
        dataset.append(features)
        labels.append(1)  # 1 for child

    print("Simulating and extracting features for negative pairs...")
    for i, (s1, s2) in enumerate(negative_pairs):
        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1}/{len(negative_pairs)} negative pairs..."
            )
        # For negative pairs, we simulate (s1 as parent, s2 as child candidate)
        # The "physics" should ideally show a different trajectory shape.
        t_coords, child_x = simulate_child_trajectory_for_pair(s1, s2)
        features = extract_features_from_trajectory(t_coords, child_x)
        dataset.append(features)
        labels.append(0)  # 0 for not-child

    X = np.array(dataset)
    y = np.array(labels)

    if len(X) == 0:
        print("No data to train on. Exiting.")
        exit()

    print(f"\nDataset shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    if len(np.unique(y)) < 2:
        print(
            "Only one class present in the dataset. Cannot train classifier."
        )
        exit()

    # Train/Test Split

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\nTraining Logistic Regression Classifier...")
    # Using a simple Logistic Regression on the 'A' parameter
    classifier = LogisticRegression(
        random_state=42, class_weight="balanced"
    )
    classifier.fit(X_train, y_train)

    print("\nEvaluating Classifier...")
    y_pred = classifier.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Optional: Visualize distribution of feature 'A' for the two classes
    if X.shape[1] > 0:  # Check if features were extracted
        plt.figure(figsize=(10, 6))
        plt.hist(
            X_train[y_train == 1, 0],
            bins=30,
            alpha=0.7,
            label="Is Child (A value)",
        )
        plt.hist(
            X_train[y_train == 0, 0],
            bins=30,
            alpha=0.7,
            label="Not Child (A value)",
        )
        plt.xlabel("Fitted Tanh Parameter 'A'")
        plt.ylabel("Frequency")
        plt.title(
            "Distribution of Feature 'A' by Class (Training Data)"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\n--- Experiment Notes ---")
    print("This is a highly experimental setup.")
    print("Key areas for improvement/exploration:")
    print(
        "1. Derivation of initial conditions & masses from embeddings (crucial!)."
    )
    print("2. Quality and source of synset embeddings.")
    print(
        "3. More sophisticated negative sampling for WordNet pairs."
    )
    print(
        "4. More features from the trajectory (e.g., fit quality, other tanh params)."
    )
    print(
        "5. Tuning of 'semantic physics' constants (Gs, default masses)."
    )
    print("6. Trying different classifiers or just thresholding 'A'.")


# --- Configuration & Global Parameters ---
SIMULATION_T_START = 0.0
SIMULATION_T_END = 1.0  # Shorter time to focus on initial swoop
SIMULATION_NUM_T_STEPS = 100
DT_SIM = (
    SIMULATION_T_END - SIMULATION_T_START
) / SIMULATION_NUM_T_STEPS

# Semantic Physics Parameters (THESE ARE HYPOTHETICAL AND NEED TUNING/DERIVATION)
G_S_GRAVITY = (
    0.05  # Universal semantic gravitational constant (example)
)
DEFAULT_M_P = 1.0  # Default parent mass
DEFAULT_M_C = 0.1  # Default child mass (lighter to be more affected)
EPSILON_SOFT = 0.01  # Softening for gravity

# Embedding model configuration
USE_GLOVE = (
    True  # Set to True to attempt using GloVe, False for random
)
# Download glove
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip
GLOVE_FILE_PATH = (
    "./glove.6B.50d.txt"  # <--- *** IMPORTANT: SET THIS PATH ***
)
EMBEDDING_DIM = 50  # Must match the dimension of your GloVe file (e.g., 50d, 100d, 200d, 300d)

# Global variable to hold loaded GloVe vectors
glove_vectors = None


# --- Function to load GloVe vectors ---
def load_glove_vectors(file_path):
    """Loads GloVe vectors from a specified file path."""
    if not os.path.exists(file_path):
        print(f"Error: GloVe file not found at {file_path}")
        print(
            "Please download GloVe vectors (e.g., glove.6B.zip from https://nlp.stanford.edu/projects/glove/)"
        )
        print("and update the GLOVE_FILE_PATH variable.")
        return None

    print(f"Loading GloVe vectors from {file_path}...")
    vectors = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                if (
                    len(vector) == EMBEDDING_DIM
                ):  # Check if dimension matches expectation
                    vectors[word] = vector
                # else:
                #     print(f"Warning: Skipping word '{word}' due to incorrect dimension ({len(vector)} != {EMBEDDING_DIM})")
        print(f"Loaded {len(vectors)} GloVe vectors.")
        # Basic check for a common word
        # if 'the' in vectors:
        #     print(f"Example vector for 'the': {vectors['the'][:5]}...")
        # else:
        #     print("Warning: Common word 'the' not found in GloVe vocabulary.")

        return vectors
    except Exception as e:
        print(f"Error loading GloVe file: {e}")
        return None


# --- Load GloVe vectors if USE_GLOVE is True ---
if USE_GLOVE:
    try:
        glove_vectors = load_glove_vectors(GLOVE_FILE_PATH)
        if glove_vectors is None:
            print(
                "Failed to load GloVe vectors. Falling back to random embeddings."
            )
            USE_GLOVE = False  # Ensure we don't try to use glove_vectors later
    except NameError:
        print(
            "load_glove_vectors function not defined yet. Will use random embeddings."
        )
        USE_GLOVE = False
        glove_vectors = None


# --- 1. Data Preparation ---


def get_synset_embedding(synset):
    """Gets a vector embedding for a synset."""
    if USE_GLOVE and glove_vectors is not None:
        vecs = []
        for lemma in synset.lemmas():
            try:
                vecs.append(
                    glove_vectors[
                        lemma.name().lower().replace("_", " ")
                    ]
                )
            except KeyError:
                try:
                    vecs.append(glove_vectors[lemma.name().lower()])
                except KeyError:
                    pass
        if vecs:
            return np.mean(vecs, axis=0)
        else:
            return np.random.rand(EMBEDDING_DIM) * 0.1
    else:
        # Consistent random embeddings for the same synset
        if not hasattr(get_synset_embedding, "random_embeddings"):
            get_synset_embedding.random_embeddings = {}
        synset_key = synset.name()
        if synset_key not in get_synset_embedding.random_embeddings:
            get_synset_embedding.random_embeddings[synset_key] = (
                np.random.rand(EMBEDDING_DIM) * 0.1
            )
        return get_synset_embedding.random_embeddings[synset_key]


def get_wordnet_pairs(num_positive=1000, num_negative_ratio=1):
    """Gets positive (hypernym-hyponym) and negative pairs from WordNet."""
    all_synsets = list(
        wn.all_synsets("n")
    )  # Focus on nouns for simplicity
    if not all_synsets:
        print(
            "Error: NLTK WordNet data might not be downloaded. Run nltk.download('wordnet')"
        )
        return [], []

    random.shuffle(all_synsets)

    positive_pairs = []
    processed_parents = set()

    for s_parent in all_synsets:
        if s_parent in processed_parents:
            continue
        hyponyms = s_parent.hyponyms()
        if hyponyms:
            for s_child in hyponyms:
                positive_pairs.append((s_parent, s_child))
                if len(positive_pairs) >= num_positive:
                    break
            processed_parents.add(
                s_parent
            )  # Avoid over-representing prolific parents
        if len(positive_pairs) >= num_positive:
            break

    num_negative = int(len(positive_pairs) * num_negative_ratio)
    negative_pairs = []
    attempts = 0
    max_attempts = num_negative * 100  # Safety break

    while (
        len(negative_pairs) < num_negative and attempts < max_attempts
    ):
        s1, s2 = random.sample(all_synsets, 2)
        attempts += 1
        if s1 == s2:
            continue
        # Check if s2 is NOT a hyponym of s1 AND s1 is NOT a hyponym of s2
        # Also ensure they are not the same path to root (e.g. great-grandparent)
        # A simple check: are they direct parent/child?
        s1_hyponyms = set(s1.hyponyms())
        s2_hyponyms = set(s2.hyponyms())
        if s2 not in s1_hyponyms and s1 not in s2_hyponyms:
            # A more robust check would trace paths, but this is a start
            common_hypernyms = set(s1.common_hypernyms(s2))
            if (
                not common_hypernyms or len(common_hypernyms) < 2
            ):  # Avoid too closely related things by chance
                negative_pairs.append((
                    s1,
                    s2,
                ))  # Order matters if we use it for parent/child roles

    print(
        f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs."
    )
    return positive_pairs, negative_pairs


# --- 2. Semantic Physics Simulation ---


def derive_initial_conditions(
    parent_synset, child_synset, parent_emb, child_emb
):
    """
    HYPOTHETICAL: Derives simulation parameters from synsets and embeddings.
    This is where most of the "semantic physics" assumptions live.
    """
    # Masses: could be based on depth, number of children/parents, or fixed
    m_p = DEFAULT_M_P
    # m_c could be inversely proportional to its "generality" or fixed
    m_c = DEFAULT_M_C * (
        1 + random.uniform(-0.05, 0.05)
    )  # Add small noise

    # Initial spatial offset: based on embedding distance?
    # For 1D simulation, we need to project. Let's use a component or distance.
    if (
        parent_emb is not None
        and child_emb is not None
        and EMBEDDING_DIM > 0
    ):
        # Use Euclidean distance for simplicity
        distance = np.linalg.norm(parent_emb - child_emb)
        # Scale distance to an initial offset. Normalize by sqrt(dim) for rough comparability
        # and adjust scaling factors (e.g., 0.5, 0.3) based on observed distances.
        initial_offset = distance / np.sqrt(
            EMBEDDING_DIM
        )  # Basic scaling
        initial_offset = np.clip(
            initial_offset * 0.5, 0.05, 0.5
        )  # Ensure some offset, clamp max

    else:  # Fallback if embeddings are None or dim is zero
        initial_offset = 0.1 + random.uniform(-0.05, 0.05)

    # Initial relative velocity:
    # Negative to induce a "swoop" towards the parent. Magnitude could vary.
    # Perhaps if they are "far" (large initial_offset), give a larger initial pull.
    initial_vx_rel = (
        -0.15 - 0.1 * initial_offset + random.uniform(-0.05, 0.05)
    )

    return m_p, m_c, initial_offset, initial_vx_rel


def simulate_child_trajectory_for_pair(parent_synset, child_synset):
    """Simulates child's path given a parent-child pair."""
    parent_emb = get_synset_embedding(parent_synset)
    child_emb = get_synset_embedding(child_synset)

    # Ensure embeddings are valid numpy arrays before passing
    if not isinstance(parent_emb, np.ndarray) or not isinstance(
        child_emb, np.ndarray
    ):
        print(
            f"Warning: Could not get valid embeddings for {parent_synset.name()} or {child_synset.name()}. Skipping pair simulation."
        )
        # Return placeholder data if simulation is skipped
        t_coords = np.linspace(
            SIMULATION_T_START,
            SIMULATION_T_END,
            SIMULATION_NUM_T_STEPS + 1,
        )
        return t_coords, np.full_like(
            t_coords, np.nan
        )  # Return NaNs for trajectory

    m_p, m_c, initial_offset, initial_vx_rel = (
        derive_initial_conditions(
            parent_synset, child_synset, parent_emb, child_emb
        )
    )

    t_coords = np.linspace(
        SIMULATION_T_START,
        SIMULATION_T_END,
        SIMULATION_NUM_T_STEPS + 1,
    )

    # Parent's mean classical path (fixed at x=0 for simplicity in this 1D relative sim)
    parent_x_mean = np.zeros_like(t_coords)
    parent_vx_mean_val = 0.0  # Parent is static in x

    # Child's classical trajectory
    child_x_classical = np.zeros_like(t_coords)
    child_vx_abs_current = (
        parent_vx_mean_val + initial_vx_rel
    )  # Absolute velocity
    child_x_classical[0] = parent_x_mean[0] + initial_offset

    for i in range(SIMULATION_NUM_T_STEPS):
        xp_i = parent_x_mean[i]
        xc_i = child_x_classical[i]

        r_x = xc_i - xp_i  # Child's position relative to parent
        dist_cubed_soft = (r_x**2 + EPSILON_SOFT**2) ** (1.5)
        accel_c_x = 0.0
        if abs(dist_cubed_soft) > 1e-12:  # Avoid division by zero
            force_g_x = (
                -G_S_GRAVITY * m_p * m_c * r_x / dist_cubed_soft
            )
            accel_c_x = force_g_x / m_c

        child_vx_abs_current += accel_c_x * DT_SIM
        child_x_classical[i + 1] = (
            child_x_classical[i] + child_vx_abs_current * DT_SIM
        )

    return t_coords, child_x_classical


# --- 3. Feature Extraction ---


def model_tanh(t, A, B, t0, C):
    """A * tanh(B * (t - t0)) + C"""
    return A * np.tanh(B * (t - t0)) + C


def extract_features_from_trajectory(t_coords, child_x_trajectory):
    """Fits tanh and extracts parameter A."""
    # Check for NaNs or Inf in trajectory
    if not np.all(np.isfinite(child_x_trajectory)):
        # print("Skipping feature extraction due to non-finite trajectory values.")
        return [
            0.0
        ]  # Return default feature if trajectory is invalid

    # Initial guesses are crucial
    mean_x = np.mean(child_x_trajectory)
    delta_x = np.max(child_x_trajectory) - np.min(child_x_trajectory)
    if delta_x < 1e-6:
        delta_x = 0.1  # Avoid zero delta_x

    # Attempt to guess direction for A
    initial_A_guess = -delta_x / 2  # Guessing a downward swoop
    # Refine guess based on start/end relative to mean
    if child_x_trajectory[-1] > child_x_trajectory[0] and abs(
        child_x_trajectory[0] - mean_x
    ) > abs(child_x_trajectory[-1] - mean_x):
        initial_A_guess = (
            delta_x / 2
        )  # upward swoop if it ends higher and starts lower rel to mean
    # If path is mostly flat, initial_A_guess might be small, which is fine.

    initial_guess = [
        initial_A_guess,  # A
        5
        / (
            t_coords[-1] - t_coords[0] + 1e-9
        ),  # B (scales with time range), avoid division by zero
        np.mean(t_coords),  # t0
        mean_x,  # C
    ]

    try:
        params, covariance = curve_fit(
            model_tanh,
            t_coords,
            child_x_trajectory,
            p0=initial_guess,
            maxfev=10000,
        )
        A_fitted = params[0]
        # Could also calculate goodness of fit (e.g., MSE or R^2) as a feature
        # y_pred = model_tanh(t_coords, *params)
        # mse = np.mean((child_x_trajectory - y_pred)**2)
        return [A_fitted]  # Return as a list of features
    except RuntimeError:
        # print("Tanh fit failed for a trajectory.")
        return [
            0.0
        ]  # Default feature if fit fails (e.g., A=0 means no clear swoop)
    except ValueError as e:
        # print(f"Value error during curve_fit: {e}. Trajectory might be problematic.")
        return [0.0]  # Default feature


# --- 4. Train Classifier ---
# (This will be done after generating all features)

# --- Main Execution ---
if __name__ == "__main__":
    print("Preparing WordNet pairs...")
    # Using small numbers for quick test, increase for real experiment
    positive_pairs, negative_pairs = get_wordnet_pairs(
        num_positive=200, num_negative_ratio=1.5
    )

    if not positive_pairs or not negative_pairs:
        print("Not enough pairs generated. Exiting.")
        exit()

    dataset = []
    labels = []

    print("Simulating and extracting features for positive pairs...")
    for i, (p_syn, c_syn) in enumerate(positive_pairs):
        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1}/{len(positive_pairs)} positive pairs..."
            )
        t_coords, child_x = simulate_child_trajectory_for_pair(
            p_syn, c_syn
        )
        # Only extract features if simulation was successful (no NaNs)
        if np.all(np.isfinite(child_x)):
            features = extract_features_from_trajectory(
                t_coords, child_x
            )
            dataset.append(features)
            labels.append(1)  # 1 for child
        # else:
        # print(f"Skipping pair ({p_syn.name()}, {c_syn.name()}) due to simulation failure.")

    print("Simulating and extracting features for negative pairs...")
    for i, (s1, s2) in enumerate(negative_pairs):
        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1}/{len(negative_pairs)} negative pairs..."
            )
        # For negative pairs, we simulate (s1 as parent, s2 as child candidate)
        # The "physics" should ideally show a different trajectory shape.
        t_coords, child_x = simulate_child_trajectory_for_pair(s1, s2)
        # Only extract features if simulation was successful (no NaNs)
        if np.all(np.isfinite(child_x)):
            features = extract_features_from_trajectory(
                t_coords, child_x
            )
            dataset.append(features)
            labels.append(0)  # 0 for not-child
        # else:
        # print(f"Skipping pair ({s1.name()}, {s2.name()}) due to simulation failure.")

    X = np.array(dataset)
    y = np.array(labels)

    if len(X) == 0:
        print("No data to train on. Exiting.")
        exit()

    print(f"\nDataset shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    if len(np.unique(y)) < 2:
        print(
            "Only one class present in the dataset. Cannot train classifier."
        )
        exit()

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\nTraining Logistic Regression Classifier...")
    # Using a simple Logistic Regression on the 'A' parameter
    classifier = LogisticRegression(
        random_state=42, class_weight="balanced"
    )
    classifier.fit(X_train, y_train)

    print("\nEvaluating Classifier...")
    y_pred = classifier.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Optional: Visualize distribution of feature 'A' for the two classes
    if (
        X.shape[1] > 0
    ):  # Check if features were extracted (should be 1 column)
        plt.figure(figsize=(10, 6))
        # Ensure there are points for each class before plotting histogram
        if np.sum(y_train == 1) > 0:
            plt.hist(
                X_train[y_train == 1, 0],
                bins=30,
                alpha=0.7,
                label="Is Child (A value)",
            )
        if np.sum(y_train == 0) > 0:
            plt.hist(
                X_train[y_train == 0, 0],
                bins=30,
                alpha=0.7,
                label="Not Child (A value)",
            )

        plt.xlabel("Fitted Tanh Parameter 'A'")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of Feature 'A' by Class (Training Data)\nUsing GloVe: {USE_GLOVE}"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\n--- Experiment Notes ---")
    print("This is a highly experimental setup.")
    print("Key areas for improvement/exploration:")
    print(
        "1. Derivation of initial conditions & masses from embeddings (crucial!)."
    )
    print(
        "2. Quality and source of synset embeddings (GloVe vs Random vs BERT etc.)."
    )
    print(
        "3. More sophisticated negative sampling for WordNet pairs."
    )
    print(
        "4. More features from the trajectory (e.g., fit quality, other tanh params, specific points)."
    )
    print(
        "5. Tuning of 'semantic physics' constants (Gs, default masses)."
    )
    print("6. Trying different classifiers or just thresholding 'A'.")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nltk.corpus import wordnet as wn
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
)  # Import Random Forest
from sklearn.metrics import classification_report, confusion_matrix


# --- [KEEP ALL THE PREVIOUS CODE FOR SIMULATION AND FEATURE EXTRACTION THE SAME] ---
# Configuration & Global Parameters
# get_synset_embedding
# get_wordnet_pairs
# derive_initial_conditions
# simulate_child_trajectory_for_pair
# model_tanh
# extract_features_from_trajectory
# ... (all the functions you had before the `if __name__ == "__main__":` block)

# --- Configuration & Global Parameters ---
SIMULATION_T_START = 0.0
SIMULATION_T_END = 1.0  # Shorter time to focus on initial swoop
SIMULATION_NUM_T_STEPS = 100
DT_SIM = (
    SIMULATION_T_END - SIMULATION_T_START
) / SIMULATION_NUM_T_STEPS

G_S_GRAVITY = 0.05
DEFAULT_M_P = 1.0
DEFAULT_M_C = 0.1
EPSILON_SOFT = 0.01

EMBEDDING_DIM = 50
USE_GLOVE = False  # Set to False by default since GloVe file likely doesn't exist
GLOVE_FILE_PATH = "./glove.6B.50d.txt"
glove_vectors = None


# (The rest of your functions: get_synset_embedding, get_wordnet_pairs, etc. remain unchanged)
def get_synset_embedding(synset):
    """Gets a vector embedding for a synset."""
    if USE_GLOVE:
        vecs = []
        for lemma in synset.lemmas():
            try:
                vecs.append(
                    glove_vectors[
                        lemma.name().lower().replace("_", " ")
                    ]
                )
            except KeyError:
                try:
                    vecs.append(glove_vectors[lemma.name().lower()])
                except KeyError:
                    pass
        if vecs:
            return np.mean(vecs, axis=0)
        else:
            return np.random.rand(EMBEDDING_DIM) * 0.1
    else:
        return np.random.rand(EMBEDDING_DIM) * 0.1


def get_wordnet_pairs(num_positive=1000, num_negative_ratio=1):
    all_synsets = list(wn.all_synsets("n"))
    if not all_synsets:
        print("Error: NLTK WordNet data might not be downloaded.")
        return [], []
    random.shuffle(all_synsets)
    positive_pairs = []
    processed_parents = set()
    for s_parent in all_synsets:
        if s_parent in processed_parents:
            continue
        hyponyms = s_parent.hyponyms()
        if hyponyms:
            for s_child in hyponyms:
                positive_pairs.append((s_parent, s_child))
                if len(positive_pairs) >= num_positive:
                    break
            processed_parents.add(s_parent)
        if len(positive_pairs) >= num_positive:
            break
    num_negative = int(len(positive_pairs) * num_negative_ratio)
    negative_pairs = []
    attempts = 0
    max_attempts = num_negative * 100
    while (
        len(negative_pairs) < num_negative and attempts < max_attempts
    ):
        s1, s2 = random.sample(all_synsets, 2)
        attempts += 1
        if s1 == s2:
            continue
        s1_hyponyms = set(s1.hyponyms())
        s2_hyponyms = set(s2.hyponyms())
        if s2 not in s1_hyponyms and s1 not in s2_hyponyms:
            common_hypernyms = set(s1.common_hypernyms(s2))
            if not common_hypernyms or len(common_hypernyms) < 2:
                negative_pairs.append((s1, s2))
    print(
        f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs."
    )
    return positive_pairs, negative_pairs


def derive_initial_conditions(
    parent_synset, child_synset, parent_emb, child_emb
):
    m_p = DEFAULT_M_P
    m_c = DEFAULT_M_C * (1 + random.uniform(-0.05, 0.05))
    if (
        parent_emb is not None
        and child_emb is not None
        and np.linalg.norm(parent_emb) > 1e-6
        and np.linalg.norm(child_emb) > 1e-6
    ):
        # Cosine distance calculation needs to handle potential zero vectors carefully if embeddings can be all zeros
        dot_product = np.dot(parent_emb, child_emb)
        norm_p = np.linalg.norm(parent_emb)
        norm_c = np.linalg.norm(child_emb)
        if norm_p * norm_c == 0:  # Avoid division by zero
            cosine_similarity = 0
        else:
            cosine_similarity = dot_product / (norm_p * norm_c)
        # initial_offset = 0.1 + 0.2 * (1 - cosine_similarity)
        euclidean_dist = np.linalg.norm(parent_emb - child_emb)
        # Normalize Euclidean distance by a rough estimate of max possible distance
        # Max L2 norm for GloVe-50 is around 5-10. Max distance could be ~20.
        # sqrt(EMBEDDING_DIM * (max_coord_val^2))
        # Let's use a simpler scaling
        initial_offset = np.clip(
            euclidean_dist
            / (np.sqrt(EMBEDDING_DIM) * 2 if USE_GLOVE else 5.0),
            0.05,
            0.3,
        )

    else:
        initial_offset = 0.1 + random.uniform(-0.05, 0.05)
    initial_vx_rel = (
        -0.15 - 0.1 * initial_offset + random.uniform(-0.05, 0.05)
    )
    return m_p, m_c, initial_offset, initial_vx_rel


def simulate_child_trajectory_for_pair(parent_synset, child_synset):
    parent_emb = get_synset_embedding(parent_synset)
    child_emb = get_synset_embedding(child_synset)
    m_p, m_c, initial_offset, initial_vx_rel = (
        derive_initial_conditions(
            parent_synset, child_synset, parent_emb, child_emb
        )
    )
    t_coords = np.linspace(
        SIMULATION_T_START,
        SIMULATION_T_END,
        SIMULATION_NUM_T_STEPS + 1,
    )
    parent_x_mean = np.zeros_like(t_coords)
    parent_vx_mean_val = 0.0
    child_x_classical = np.zeros_like(t_coords)
    child_vx_abs_current = parent_vx_mean_val + initial_vx_rel
    child_x_classical[0] = parent_x_mean[0] + initial_offset
    for i in range(SIMULATION_NUM_T_STEPS):
        xp_i = parent_x_mean[i]
        xc_i = child_x_classical[i]
        r_x = xc_i - xp_i
        dist_cubed_soft = (r_x**2 + EPSILON_SOFT**2) ** (1.5)
        accel_c_x = 0.0
        if abs(dist_cubed_soft) > 1e-12:
            force_g_x = (
                -G_S_GRAVITY * m_p * m_c * r_x / dist_cubed_soft
            )
            accel_c_x = force_g_x / m_c
        child_vx_abs_current += accel_c_x * DT_SIM
        child_x_classical[i + 1] = (
            child_x_classical[i] + child_vx_abs_current * DT_SIM
        )
    return t_coords, child_x_classical


def model_tanh(t, A, B, t0, C):
    return A * np.tanh(B * (t - t0)) + C


def extract_features_from_trajectory(t_coords, child_x_trajectory):
    mean_x = np.mean(child_x_trajectory)
    min_x, max_x = (
        np.min(child_x_trajectory),
        np.max(child_x_trajectory),
    )
    delta_x = max_x - min_x
    if delta_x < 1e-6:
        delta_x = 0.1
    initial_A_guess = -delta_x / 2
    if (
        len(child_x_trajectory) > 1
        and child_x_trajectory[-1] > child_x_trajectory[0]
        and abs(child_x_trajectory[0] - mean_x)
        > abs(child_x_trajectory[-1] - mean_x)
    ):
        initial_A_guess = delta_x / 2
    initial_guess = [
        initial_A_guess,
        5 / (t_coords[-1] - t_coords[0] + 1e-9),
        np.mean(t_coords),
        mean_x,
    ]
    try:
        params, covariance = curve_fit(
            model_tanh,
            t_coords,
            child_x_trajectory,
            p0=initial_guess,
            maxfev=10000,
        )
        A_fitted = params[0]
        return [A_fitted]
    except RuntimeError:
        return [0.0]
    except (
        ValueError
    ):  # Can happen if inputs to curve_fit are bad (e.g. all NaNs)
        return [0.0]


# --- Main Execution ---
if __name__ == "__main__":
    # --- [Data generation and feature extraction code from your previous script] ---
    print("Preparing WordNet pairs...")
    positive_pairs, negative_pairs = get_wordnet_pairs(
        num_positive=500, num_negative_ratio=1.5
    )  # Reduced for faster test

    if not positive_pairs or not negative_pairs:
        print("Not enough pairs generated. Exiting.")
        exit()

    dataset = []
    labels = []

    print("Simulating and extracting features for positive pairs...")
    for i, (p_syn, c_syn) in enumerate(positive_pairs):
        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1}/{len(positive_pairs)} positive pairs..."
            )
        t_coords, child_x = simulate_child_trajectory_for_pair(
            p_syn, c_syn
        )
        features = extract_features_from_trajectory(t_coords, child_x)
        dataset.append(features)
        labels.append(1)

    print("Simulating and extracting features for negative pairs...")
    for i, (s1, s2) in enumerate(negative_pairs):
        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1}/{len(negative_pairs)} negative pairs..."
            )
        t_coords, child_x = simulate_child_trajectory_for_pair(
            s1, s2
        )  # s1 as parent, s2 as child candidate
        features = extract_features_from_trajectory(t_coords, child_x)
        dataset.append(features)
        labels.append(0)

    X = np.array(dataset)
    y = np.array(labels)

    if len(X) == 0:
        print("No data to train on. Exiting.")
        exit()

    print(f"\nDataset shape: {X.shape}, Labels shape: {y.shape}")
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")

    if len(unique_labels) < 2:
        print(
            "Only one class present in the dataset. Cannot train classifier."
        )
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- Logistic Regression (as before, for baseline) ---
    print("\n--- Logistic Regression Classifier ---")
    log_reg_classifier = LogisticRegression(
        random_state=42, class_weight="balanced", solver="liblinear"
    )
    log_reg_classifier.fit(X_train, y_train)
    y_pred_log_reg = log_reg_classifier.predict(X_test)

    print("Confusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_test, y_pred_log_reg))
    print("\nClassification Report (Logistic Regression):")
    print(
        classification_report(y_test, y_pred_log_reg, zero_division=0)
    )

    # --- Random Forest Classifier ---
    print("\n--- Random Forest Classifier ---")
    rf_classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)

    print("Confusion Matrix (Random Forest):")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    # --- [Optional: Feature Distribution Plot code from your previous script] ---
    if X.shape[1] > 0:
        plt.figure(figsize=(10, 6))
        # Use training data for plotting feature distribution
        plt.hist(
            X_train[y_train == 1, 0],
            bins=30,
            alpha=0.7,
            label="Is Child (A value)",
        )
        plt.hist(
            X_train[y_train == 0, 0],
            bins=30,
            alpha=0.7,
            label="Not Child (A value)",
        )
        plt.xlabel("Fitted Tanh Parameter 'A'")
        plt.ylabel("Frequency")
        plt.title(
            "Distribution of Feature 'A' by Class (Training Data)"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\n--- Experiment Notes (Continued) ---")
    print("Comparing Random Forest to Logistic Regression.")
    print(
        "Key areas for improvement remain the same: initial conditions, embeddings, features, physics constants."
    )


def extract_features_from_trajectory(t_coords, child_x_trajectory):
    features = [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # A, B, t0, C, MSE (default for fit failure)
    # Using 1.0 for MSE failure as high error

    # Trajectory summary stats (calculated before fit attempt)
    min_x_child = np.min(child_x_trajectory)
    t_at_min_x = t_coords[np.argmin(child_x_trajectory)]
    mean_x_child = np.mean(child_x_trajectory)
    # Add these as features:
    # features.extend([min_x_child, t_at_min_x, mean_x_child]) # Example

    try:
        # Initial guesses (as before, or improved)
        delta_x = np.max(child_x_trajectory) - np.min(
            child_x_trajectory
        )
        if delta_x < 1e-6:
            delta_x = 0.1
        initial_A_guess = -delta_x / 2
        if (
            len(child_x_trajectory) > 1
            and child_x_trajectory[-1] > child_x_trajectory[0]
            and abs(child_x_trajectory[0] - mean_x_child)
            > abs(child_x_trajectory[-1] - mean_x_child)
        ):
            initial_A_guess = delta_x / 2

        initial_guess = [
            initial_A_guess,
            5 / (t_coords[-1] - t_coords[0] + 1e-9),  # B
            np.mean(t_coords),  # t0
            mean_x_child,  # C
        ]
        params, covariance = curve_fit(
            model_tanh,
            t_coords,
            child_x_trajectory,
            p0=initial_guess,
            maxfev=10000,
        )

        y_pred_tanh = model_tanh(t_coords, *params)
        mse = np.mean((child_x_trajectory - y_pred_tanh) ** 2)

        features = [params[0], params[1], params[2], params[3], mse]
        features.extend([
            min_x_child,
            t_at_min_x,
            mean_x_child,
        ])  # Add summary stats

    except RuntimeError:
        # print("Tanh fit failed. Using default/summary features.")
        # Use only summary stats if fit fails, or keep defaults for fit params
        features.extend([min_x_child, t_at_min_x, mean_x_child])
        # Ensure features list has the correct length if fit fails
        while len(features) < 8:  # 5 fit params + 3 summary
            features.append(0.0)  # Pad with neutral values
        features = features[:8]  # Ensure correct length

    except ValueError:  # Can happen if inputs to curve_fit are bad
        # print("ValueError during Tanh fit. Using default/summary features.")
        features.extend([min_x_child, t_at_min_x, mean_x_child])
        while len(features) < 8:
            features.append(0.0)
        features = features[:8]

    return features


# prompt: Call extract features from traj

import numpy as np

if dataset:
    print(
        "\nDemonstrating a call to extract_features_from_trajectory with a sample simulated pair:"
    )
    if positive_pairs:
        sample_parent, sample_child = positive_pairs[0]
        try:
            sample_t_coords, sample_child_x_traj = (
                simulate_child_trajectory_for_pair(
                    sample_parent, sample_child
                )
            )

            if np.all(np.isfinite(sample_child_x_traj)):
                sample_features = extract_features_from_trajectory(
                    sample_t_coords, sample_child_x_traj
                )
                print(
                    f"Features extracted for pair ({sample_parent.name()}, {sample_child.name()}): {sample_features}"
                )
            else:
                print(
                    f"Simulation for sample pair ({sample_parent.name()}, {sample_child.name()}) failed, cannot extract features."
                )

        except Exception as e:
            print(
                f"An error occurred during sample simulation/feature extraction: {e}"
            )
    else:
        print(
            "No positive pairs were generated, cannot demonstrate feature extraction."
        )

else:
    print(
        "The 'dataset' list is empty. Please run the previous cells to generate data."
    )

from IPython import get_ipython
from IPython.display import display

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
import pickle

# Assume embeddings are loaded and preprocessed elsewhere
# Example dummy embeddings: Replace with your actual data loading
# Ensure parent_embeddings and child_embeddings are aligned such that
# parent_embeddings[i] corresponds to child_embeddings[i].
num_samples = 200
embedding_dim = 5
parent_embeddings = np.random.rand(num_samples, embedding_dim) * 2 - 1
child_embeddings = np.random.rand(num_samples, embedding_dim) * 2 - 1
# Simulate t_parent and t_child where t_child > t_parent
t_parent = np.random.rand(num_samples) * 5
t_child = t_parent + np.random.rand(num_samples) * 2 + 0.1
# Simulate the target delta_t values
target_delta_t = t_child - t_parent  # This is the target variable

# %%
# --- Feature Engineering ---

# 1. Basic Embedding Features
features = pd.DataFrame()
features["parent_norm"] = np.linalg.norm(parent_embeddings, axis=1)
features["child_norm"] = np.linalg.norm(child_embeddings, axis=1)
features["dot_product"] = np.sum(
    parent_embeddings * child_embeddings, axis=1
)
features["cosine_similarity"] = features["dot_product"] / (
    features["parent_norm"] * features["child_norm"] + 1e-8
)  # Add epsilon for stability

# 2. Time Features
features["t_parent"] = t_parent
features["t_child"] = t_child
features["time_diff_initial"] = (
    features["t_child"] - features["t_parent"]
)  # This is the TARGET, not a feature for prediction!
# Remove the target from features for training
target = features["time_diff_initial"]
features = features.drop("time_diff_initial", axis=1)


# Function to fit tanh curve to trajectory (simplified for 1D spatial coord)
# Assuming 'trajectories' are lists of (t, x) points or similar structure
# For this example, let's generate dummy trajectories or assume they come from data
def tanh_func(t, A, B, t0, C):
    return A * np.tanh(B * (t - t0)) + C


# Placeholder for trajectory data - REPLACE WITH YOUR ACTUAL TRAJECTORY DATA
# This must be a list where trajectories[i] is the sequence of (t, x) points
# corresponding to the parent/child pair i.
# Example dummy trajectories (a straight line with some noise)
dummy_trajectories = []
for i in range(num_samples):
    num_points = np.random.randint(5, 20)  # Varying number of points
    t_traj = np.linspace(t_parent[i], t_child[i], num_points)
    # Simple linear path + noise
    x_traj = (
        np.linspace(0, 1, num_points)
        + np.random.randn(num_points) * 0.1
    )
    dummy_trajectories.append(list(zip(t_traj, x_traj)))

# 3. Tanh Fit Parameters and MSE (Requires trajectory data)
tanh_params_list = []
tanh_mse_list = []

for traj in dummy_trajectories:
    if len(traj) < 4:  # Need at least 4 points to fit 4 parameters
        tanh_params_list.append([np.nan] * 4)
        tanh_mse_list.append(np.nan)
        continue

    t_data = np.array([p[0] for p in traj])
    x_data = np.array([p[1] for p in traj])

    # Provide initial guess for parameters
    try:
        # Simple initial guess: A ~ range of x, B ~ 1/(range of t), t0 ~ mid-t, C ~ mid-x
        A_guess = (np.max(x_data) - np.min(x_data)) / 2
        B_guess = 1.0 / (t_data[-1] - t_data[0] + 1e-8)
        t0_guess = (t_data[0] + t_data[-1]) / 2
        C_guess = (np.max(x_data) + np.min(x_data)) / 2
        initial_guess = [A_guess, B_guess, t0_guess, C_guess]

        # Set bounds for parameters if possible (e.g., B > 0)
        bounds = (
            [-np.inf, 0, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        )

        params, _ = curve_fit(
            tanh_func,
            t_data,
            x_data,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
        )
        tanh_params_list.append(params.tolist())
        fitted_curve = tanh_func(t_data, *params)
        mse = mean_squared_error(x_data, fitted_curve)
        tanh_mse_list.append(mse)
    except Exception as e:
        print(f"Tanh fit failed for trajectory: {e}")
        tanh_params_list.append([np.nan] * 4)
        tanh_mse_list.append(np.nan)

tanh_params_df = pd.DataFrame(
    tanh_params_list,
    columns=["tanh_A", "tanh_B", "tanh_t0", "tanh_C"],
)
features = pd.concat([features, tanh_params_df], axis=1)
features["tanh_mse"] = tanh_mse_list

# 4. Trajectory Summary Statistics (Requires trajectory data)
# Assuming 'trajectories' are lists of (t, x) points
traj_length = []
traj_duration = []
traj_spatial_range = []
traj_mean_speed = []  # Simple straight-line speed approximation

for traj in dummy_trajectories:
    if len(traj) < 2:
        traj_length.append(np.nan)
        traj_duration.append(np.nan)
        traj_spatial_range.append(np.nan)
        traj_mean_speed.append(np.nan)
        continue

    t_data = np.array([p[0] for p in traj])
    x_data = np.array([p[1] for p in traj])

    # Total distance along path (approximate)
    path_dist = np.sum(
        np.sqrt(np.diff(t_data) ** 2 + np.diff(x_data) ** 2)
    )
    traj_length.append(path_dist)

    # Total duration
    duration = t_data[-1] - t_data[0]
    traj_duration.append(duration)

    # Spatial range
    spatial_range = np.max(x_data) - np.min(x_data)
    traj_spatial_range.append(spatial_range)

    # Straight-line speed between start and end
    if duration > 1e-8:
        mean_speed = (
            np.sqrt(
                (t_data[-1] - t_data[0]) ** 2
                + (x_data[-1] - x_data[0]) ** 2
            )
            / duration
        )
        traj_mean_speed.append(mean_speed)
    else:
        traj_mean_speed.append(np.nan)


features["traj_length"] = traj_length
features["traj_duration"] = traj_duration
features["traj_spatial_range"] = traj_spatial_range
features["traj_mean_speed"] = traj_mean_speed

# Handle NaNs created by failed fits or insufficient trajectory points
features = features.fillna(
    features.median()
)  # Simple imputation: fill NaNs with median

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# --- Scaling Features ---
# It's good practice to scale features for many models, including tree-based ones sometimes,
# although less critical than for linear models or neural networks.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep column names if needed later, though not strictly necessary for RF
X_train_scaled_df = pd.DataFrame(
    X_train_scaled, columns=X_train.columns
)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# %%
# --- Tune Random Forest ---

# Define the parameter space for RandomizedSearchCV
param_dist = {
    "n_estimators": randint(50, 500),  # Number of trees in the forest
    "max_features": [
        "auto",
        "sqrt",
        "log2",
    ],  # Number of features to consider at every split
    "max_depth": randint(10, 110),  # Maximum number of levels in tree
    "min_samples_split": randint(
        2, 20
    ),  # Minimum number of samples required to split an internal node
    "min_samples_leaf": randint(
        1, 20
    ),  # Minimum number of samples required to be at a leaf node
    "bootstrap": [
        True,
        False,
    ],  # Whether bootstrap samples are used when building trees
}

# Create a Random Forest Regressor model
rf = RandomForestRegressor(random_state=42)

# Set up RandomizedSearchCV
# n_iter: Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.
# cv: Number of folds in cross-validation.
# scoring: Metric to evaluate the model (neg_mean_squared_error is common for regression tuning)
# n_jobs: Number of cores to use (-1 means all available cores)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

# Fit RandomizedSearchCV to the training data
print("Starting Randomized Search for Random Forest tuning...")
random_search.fit(X_train_scaled, y_train)

print("\nRandomized Search complete.")
print("Best parameters found: ", random_search.best_params_)
print("Best negative MSE score: ", random_search.best_score_)

# Get the best model
best_rf_model = random_search.best_estimator_

# --- Evaluate the best model ---
y_pred = best_rf_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(
    f"\nEvaluation of the best Random Forest model on the test set:"
)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Feature Importance (using the best model)
importances = best_rf_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
})
feature_importance_df = feature_importance_df.sort_values(
    "importance", ascending=False
)

print("\nTop 10 Features by Importance:")
print(feature_importance_df.head(10))


# %%
# --- Refine Initial Conditions Derivation ---

# This section focuses on the logic for derive_initial_conditions,
# suggesting ways to make initial_offset and initial_vx_rel more semantically meaningful
# based on embeddings.


def derive_initial_conditions_refined(
    parent_emb, child_emb, t_parent, t_child
):
    """
    Derives initial spatial position and relative velocity based on semantic embeddings.
    Analogy to initial conditions in physics, using embedding relationships.

    Args:
        parent_emb (np.ndarray): Embedding vector for the parent concept.
        child_emb (np.ndarray): Embedding vector for the child concept.
        t_parent (float): Semantic time of the parent event.
        t_child (float): Semantic time of the child event.

    Returns:
        tuple: (initial_offset_x, initial_vx_rel_x)
               initial_offset_x (float): Initial spatial offset (x_child - x_parent) at t_parent.
               initial_vx_rel_x (float): Initial spatial velocity of child relative to parent.
    """
    # Ensure embeddings are numpy arrays
    parent_emb = np.asarray(parent_emb)
    child_emb = np.asarray(child_emb)

    # Calculate the difference vector
    diff_vector = child_emb - parent_emb
    diff_norm = np.linalg.norm(diff_vector)

    # 1. Initial Spatial Offset (x_child - x_parent)
    # The magnitude of the difference vector could relate to the initial spatial separation.
    # Let's use the norm as a proxy for spatial distance in the semantic space at t_parent.
    initial_offset_x = (
        diff_norm  # Or a scaled version: diff_norm * some_scale
    )

    # 2. Initial Relative Velocity (vx_child - vx_parent)
    # This is more speculative and depends on the semantic space structure.
    # Idea: Project the child embedding onto the parent-child difference vector.
    # This projection magnitude/direction could represent the child's 'momentum'
    # or 'tendency' away from/towards the parent in the direction of their difference.

    if diff_norm > 1e-8:
        # Normalize the difference vector
        diff_vector_normalized = diff_vector / diff_norm

        # Project child embedding onto the normalized difference vector
        projection_magnitude = np.dot(
            child_emb, diff_vector_normalized
        )

        # The projection magnitude could be used as a basis for initial_vx_rel_x.
        # A positive projection means child embedding points somewhat in the direction
        # from parent to child, suggesting 'movement' in that direction.
        # A negative projection means child embedding points against the direction,
        # suggesting 'movement' towards the parent or away in the opposite direction.

        # Let's use the projection directly, or a scaled version.
        initial_vx_rel_x = projection_magnitude  # Or projection_magnitude * some_velocity_scale

        # Alternative idea: Use the *change* in the projection over semantic time
        # This requires knowing embeddings at multiple time points, which we don't have here.
        # The current approach is simpler: infer 'velocity' from static embedding relationship.

    else:  # Parent and child embeddings are identical
        initial_offset_x = 0.0
        initial_vx_rel_x = 0.0  # No difference, no relative velocity

    # Additional considerations (could be added as features or used in derivation):
    # - Cosine similarity: Highly similar embeddings might imply a small initial offset and low relative velocity.
    # - Time difference (t_child - t_parent): A large time difference might suggest a trajectory with more 'time' to evolve, potentially affecting initial conditions indirectly if they were derived from a dynamic process.

    return initial_offset_x, initial_vx_rel_x


# --- Example Usage of Refined Initial Conditions Derivation ---
# (This part would typically be integrated into your simulation loop or feature generation)

sample_index = 0  # Let's look at the first sample

parent_emb_sample = parent_embeddings[sample_index]
child_emb_sample = child_embeddings[sample_index]
t_parent_sample = t_parent[sample_index]
t_child_sample = t_child[sample_index]

initial_offset, initial_vx_rel = derive_initial_conditions_refined(
    parent_emb_sample,
    child_emb_sample,
    t_parent_sample,
    t_child_sample,
)

print(f"\nRefined Initial Conditions for sample {sample_index}:")
print(f"Parent Embedding: {parent_emb_sample}")
print(f"Child Embedding: {child_emb_sample}")
print(
    f"t_parent: {t_parent_sample:.2f}, t_child: {t_child_sample:.2f}"
)
print(
    f"Derived Initial Spatial Offset (x_child - x_parent): {initial_offset:.4f}"
)
print(
    f"Derived Initial Relative Velocity (vx_child - vx_parent): {initial_vx_rel:.4f}"
)
# %% [markdown]
# Integration with Classical Simulation
# --------------------------------
# Replace fixed initialization with dynamic calculation per sample:
# - Use derive_initial_conditions_refined() for each parent-child pair
# - Parameters: parent_embeddings[i], child_embeddings[i], t_parent[i], t_child[i]
# - Apply derived values as initial conditions in simulation loop

# Path Integral Implementation
# ---------------------------
# Classical path serves as mean trajectory:
# - Calculate using refined initial conditions
# - Add Brownian bridge fluctuations around mean path
# - Scale perturbation_strength based on embedding similarity:
#   * Higher similarity → Lower fluctuations
#   * Lower similarity → Higher fluctuations

# ---------------------

# %%
# OPTION 1: Use California Housing Dataset (built into scikit-learn)
print("Loading California Housing dataset...")

# Load the dataset
housing = fetch_california_housing()
X_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
y_housing = housing.target

print(f"Dataset shape: {X_housing.shape}")
print(f"Features: {list(X_housing.columns)}")
print(f"Target: House values in hundreds of thousands of dollars")
print("\nFirst few rows:")
print(X_housing.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# %%
# OPTION 2: Use Boston Housing from Hugging Face (alternative)
# Uncomment this section if you prefer to use a Hugging Face dataset

# try:
#     print("Loading Boston Housing dataset from Hugging Face...")
#     dataset = load_dataset("mstz/boston_housing")
#
#     # Convert to pandas DataFrame
#     train_df = dataset['train'].to_pandas()
#
#     # Separate features and target
#     feature_cols = [col for col in train_df.columns if col != 'medv']
#     X_housing = train_df[feature_cols]
#     y_housing = train_df['medv']
#
#     print(f"Dataset shape: {X_housing.shape}")
#     print(f"Features: {list(X_housing.columns)}")
#
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_housing, y_housing, test_size=0.2, random_state=42
#     )
#
# except Exception as e:
#     print(f"Error loading from Hugging Face: {e}")
#     print("Falling back to California Housing dataset...")
#     # Use the California Housing code above as fallback

# %%
# Train Random Forest with hyperparameter tuning
print("\nSetting up hyperparameter tuning...")

# Define parameter grid for RandomizedSearchCV
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, 50, 100, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

# Create Random Forest model
rf = RandomForestRegressor(random_state=42)

# Perform randomized search
print("Performing hyperparameter tuning...")
random_search = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=50,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

# Fit the random search
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(
    f"Best cross-validation score: {-random_search.best_score_:.4f}"
)

# %% ======================================================
# Train final model with best parameters
print("\nTraining final model with best parameters...")
best_rf_model = random_search.best_estimator_

# Make predictions
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# %%
# Feature importance analysis
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_rf_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# %%
# OPTION 3: Load your own CSV data
# Uncomment and modify this section if you have your own dataset

# print("\nLoading custom dataset from CSV...")
# # Replace 'your_dataset.csv' with your actual file path
# df = pd.read_csv('your_dataset.csv')
#
# # Specify your target column name
# target_column = 'target'  # Replace with your actual target column name
#
# # Separate features and target
# X_custom = df.drop(columns=[target_column])
# y_custom = df[target_column]
#
# # Handle categorical variables if needed
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# for column in X_custom.select_dtypes(include=['object']).columns:
#     X_custom[column] = le.fit_transform(X_custom[column])
#
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(
#     X_custom, y_custom, test_size=0.2, random_state=42
# )
#
# # Continue with the same training process as above...

# %%
# Test on new data from Hugging Face (using the trained model)
try:
    print("\nTesting trained model on NYC taxi data sample...")

    # Load NYC taxi dataset sample
    taxi_dataset = load_dataset("nyc_taxi", split="train[:100]")
    taxi_df = taxi_dataset.to_pandas()

    # Create features that match our housing model (this is just for demonstration)
    # In practice, you'd want to use a dataset with similar features to your training data
    print(
        "Note: This is just a demonstration - taxi data features don't match housing features"
    )
    print(
        "In practice, use datasets with similar feature types for meaningful predictions"
    )

    # Create dummy features matching our housing model
    taxi_features = pd.DataFrame()
    for feature in X_train.columns:
        taxi_features[feature] = np.random.normal(
            X_train[feature].mean(),
            X_train[feature].std(),
            len(taxi_df),
        )

    # Make predictions
    taxi_predictions = best_rf_model.predict(taxi_features)

    print(
        f"Made {len(taxi_predictions)} predictions on taxi data sample"
    )
    print(f"Sample predictions: {taxi_predictions[:5]}")

except Exception as e:
    print(f"Error with taxi data demonstration: {e}")

print("\n=== Training Complete ===")
print("Your model is now trained on real data!")
print(
    "To use your own dataset, modify the code in the OPTION 3 section above."
)

# Embedding model configuration
USE_GLOVE = (
    True  # Set to True to attempt using GloVe, False for random
)
# Download glove
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip
GLOVE_FILE_PATH = (
    "./glove.6B.50d.txt"  # <--- *** IMPORTANT: SET THIS PATH ***
)
EMBEDDING_DIM = 50  # Must match the dimension of your GloVe file (e.g., 50d, 100d, 200d, 300d)

# Global variable to hold loaded GloVe vectors
glove_vectors = None

# --- Load GloVe vectors if USE_GLOVE is True ---
if USE_GLOVE:
    try:
        glove_vectors = load_glove_vectors(GLOVE_FILE_PATH)
        if glove_vectors is None:
            print(
                "Failed to load GloVe vectors. Falling back to random embeddings."
            )
            USE_GLOVE = False  # Ensure we don't try to use glove_vectors later
    except NameError:
        print(
            "load_glove_vectors function not defined yet. Will use random embeddings."
        )
        USE_GLOVE = False
        glove_vectors = None


def get_synset_embedding(synset):
    """Gets a vector embedding for a synset."""
    # Use consistent random embeddings for the same synset
    if not hasattr(get_synset_embedding, "random_embeddings"):
        get_synset_embedding.random_embeddings = {}
    synset_key = synset.name()
    if synset_key not in get_synset_embedding.random_embeddings:
        get_synset_embedding.random_embeddings[synset_key] = (
            np.random.rand(EMBEDDING_DIM) * 0.1
        )
    return get_synset_embedding.random_embeddings[synset_key]
