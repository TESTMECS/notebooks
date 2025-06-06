"""
Latent Minkowski Swarm Optimizer (LMSO) is a swarm optimization algorithm that uses
Minkowski space to optimize functions with multiple local minima. It is designed to
efficiently explore the search space and find the global minimum. The algorithm uses
physics-based causal connections to guide the swarm towards the global minimum.
"""

import math
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sklearn.cluster import DBSCAN

console = Console()


# --- Neural Physics Module ---
class CausalConnectionPredictor(nn.Module):
    """Neural network to predict causal connections in spacetime"""

    def __init__(self, input_dim=6, hidden_dims=[64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        # Output layer (probability of causal connection)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def physics_loss(model_edges, node_positions):
    """
    Calculate energy-based physics loss for a set of edges.
    Penalizes high-energy or non-causal (spacelike) connections.
    """
    if not model_edges:
        return 0.0

    loss = 0.0
    for u, v in model_edges:
        if u in node_positions and v in node_positions:
            p1, p2 = node_positions[u], node_positions[v]
            dx = p2[0] - p1[0]  # x_emb difference
            dy = p2[1] - p1[1]  # y_emb difference
            dt = p2[2] - p1[2]  # time difference

            ds2 = -(dt**2) + dx**2 + dy**2  # Minkowski interval

            if ds2 < -1e-2:  # Timelike (causal)
                energy = np.sqrt(-ds2)
                loss += energy  # Encourage low-energy causal links
            else:  # Spacelike (non-causal)
                loss += 10.0  # Heavy penalty for non-causal connections

    return loss / len(model_edges)


def extract_edge_features(event1, event2):
    """Extract spacetime features for two events"""
    dx = event2.x_emb - event1.x_emb
    dy = event2.y_emb - event1.y_emb
    dt = event2.t - event1.t
    ds2 = -(dt**2) + dx**2 + dy**2

    spatial_dist = np.sqrt(dx**2 + dy**2)
    is_timelike = 1.0 if ds2 < 0 else 0.0

    return torch.FloatTensor([dx, dy, dt, ds2, spatial_dist, is_timelike])


def interaction_step(space, max_merges=3):
    """
    Physics-inspired graph evolution: merge low-energy connections
    and redistribute causal structure for efficiency
    """
    events = list(space.events.values())
    if len(events) < 2:
        return

    # Find potential merges based on low spacetime energy
    merge_candidates = []

    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events[i + 1 :], i + 1):
            dx = e2.x_emb - e1.x_emb
            dy = e2.y_emb - e1.y_emb
            dt = e2.t - e1.t
            ds2 = -(dt**2) + dx**2 + dy**2

            if ds2 < -1e-2:  # Timelike connection
                energy = np.sqrt(-ds2)
                merge_candidates.append((energy, e1, e2))

    # Sort by energy and merge the lowest-energy pairs
    merge_candidates.sort(key=lambda x: x[0])

    merged_count = 0
    for energy, e1, e2 in merge_candidates[:max_merges]:
        if e1.id in space.events and e2.id in space.events:
            # Create merged event
            merged_solution = [
                (e1.solution[0] + e2.solution[0]) / 2,
                (e1.solution[1] + e2.solution[1]) / 2,
            ]
            merged_fitness = min(e1.fitness, e2.fitness)  # Take better fitness
            merged_t = (e1.t + e2.t) / 2
            merged_x_emb = (e1.x_emb + e2.x_emb) / 2
            merged_y_emb = (e1.y_emb + e2.y_emb) / 2

            # Remove old events
            del space.events[e1.id]
            del space.events[e2.id]

            # Add merged event
            merged_event = space.add_event(
                merged_solution, merged_fitness, merged_t, merged_x_emb, merged_y_emb
            )

            # Update global best if needed
            if merged_fitness < space.global_best_event.fitness:
                space.global_best_event = merged_event

            merged_count += 1

    if merged_count > 0:
        console.print(
            f"[dim cyan]⚛️  Physics merged {merged_count} low-energy event pairs[/dim cyan]"
        )


# Global causal predictor (will be trained during optimization)
causal_predictor = None


# --- Objective Function ---
def sphere_function(x, y):
    return x**2 + y**2  # Minimize this


def rastrigin_function(x, y, A=10):
    return (
        A * 2
        + (x**2 - A * math.cos(2 * math.pi * x))
        + (y**2 - A * math.cos(2 * math.pi * y))
    )


# --- Event & Minkowski Space (Simplified from previous notebook) ---
class Event:
    def __init__(self, id, solution, fitness, t, x_emb, y_emb):
        self.id = id
        self.solution = solution  # (x, y) in problem space
        self.fitness = fitness
        self.t = float(t)
        self.x_emb = float(x_emb)  # Embedding coordinate
        self.y_emb = float(y_emb)  # Embedding coordinate

    def __repr__(self):
        return f"E({self.id}, S=({self.solution[0]:.2f},{self.solution[1]:.2f}), F={self.fitness:.2f}, t={self.t:.2f}, emb=({self.x_emb:.2f},{self.y_emb:.2f}))"  # noqa: E501

    def __lt__(self, other):  # For sorting if needed
        return self.t < other.t


class MinkowskiSpace:
    def __init__(self):
        self.events = {}  # id -> Event
        self.event_counter = 0
        self.global_best_event = None

    def add_event(self, solution, fitness, t_initial, x_emb_initial, y_emb_initial):
        event_id = self.event_counter
        self.event_counter += 1
        event = Event(
            event_id, solution, fitness, t_initial, x_emb_initial, y_emb_initial
        )
        self.events[event_id] = event
        if self.global_best_event is None or fitness < self.global_best_event.fitness:
            self.global_best_event = event
        return event

    def get_event(self, event_id):
        return self.events.get(event_id)

    def get_recent_events(self, count=5):
        # Simplistic: just get last 'count' events by ID
        # A real one might get recent by 't'
        if not self.events:
            return []
        sorted_events = sorted(self.events.values(), key=lambda e: e.id, reverse=True)
        return sorted_events[:count]

    def calculate_spacetime_interval_sq(self, e1, e2):
        dt = e1.t - e2.t
        dx_emb = e1.x_emb - e2.x_emb
        dy_emb = e1.y_emb - e2.y_emb
        return -(dt**2) + dx_emb**2 + dy_emb**2

    def is_in_future_light_cone(self, parent_event, child_event, epsilon=1e-3):
        global causal_predictor

        if child_event.t <= parent_event.t:
            return False

        # If we have a trained causal predictor, use it
        if causal_predictor is not None:
            try:
                features = extract_edge_features(parent_event, child_event)
                with torch.no_grad():
                    causal_prob = causal_predictor(features.unsqueeze(0)).item()
                return causal_prob > 0.5
            except:
                # Fallback to physics check if neural network fails
                pass

        # Fallback: traditional physics check
        interval_sq = self.calculate_spacetime_interval_sq(parent_event, child_event)
        return interval_sq <= epsilon  # Timelike or lightlike

    def get_causal_parents(self, event, max_count=10):
        """Get potential causal parents using learned physics"""
        potential_parents = []

        for parent_id, parent_event in self.events.items():
            if parent_id != event.id and self.is_in_future_light_cone(
                parent_event, event
            ):
                # Calculate physics-based score
                dx = event.x_emb - parent_event.x_emb
                dy = event.y_emb - parent_event.y_emb
                dt = event.t - parent_event.t
                ds2 = -(dt**2) + dx**2 + dy**2

                if ds2 < 0:  # Timelike
                    energy = np.sqrt(-ds2)
                    fitness_improvement = max(0, parent_event.fitness - event.fitness)
                    score = fitness_improvement / (
                        energy + 1e-6
                    )  # Higher score = better causal connection
                    potential_parents.append((score, parent_event))

        # Sort by score and return top candidates
        potential_parents.sort(key=lambda x: x[0], reverse=True)
        return [parent for score, parent in potential_parents[:max_count]]


# --- "Magic Box" Placeholders ---
def initial_embed_solution(solution, fitness, iteration, space):
    """Randomly assigns initial embedding coords for a new solution."""
    # t is based on iteration
    t_initial = float(iteration)
    # embedding coords initially random or based on solution for diversity
    x_emb_initial = random.uniform(-1, 1)  # Or map solution.x to this range
    y_emb_initial = random.uniform(-1, 1)  # Or map solution.y to this range
    return space.add_event(solution, fitness, t_initial, x_emb_initial, y_emb_initial)


def project_until_convergence(
    pairs: np.ndarray,
    spatial: np.ndarray,
    time_vec: np.ndarray,
    eps1: float = 1e-5,
    max_passes: int = 10000,
) -> int:
    """Deterministically enforce `Δt_parent-child > ||Δx||` by shifting *only* `t`.

    Modifies time_vec in place.
    Assumes pairs are [child_idx, parent_idx].
    We want t_parent > t_child for the paper's causality interpretation.
    The function will adjust t_parent and t_child to satisfy:
    (t_parent - t_child) > ||spatial_parent - spatial_child|| + eps1

    Returns the number of full passes executed until convergence.
    """

    # Create a beautiful info panel
    info_panel = Panel.fit(
        f"[bold cyan]Starting Convergence Algorithm[/bold cyan]\n"
        f"[green]Pairs:[/green] {len(pairs)}\n"
        f"[green]Nodes:[/green] {len(time_vec)}\n"
        f"[green]Max Passes:[/green] {max_passes}\n"
        f"[green]Epsilon:[/green] {eps1:.2e}",
        title="🚀 Convergence Setup",
        border_style="cyan",
    )
    console.print(info_panel)

    # Initial statistics table
    stats_table = Table(title="📊 Initial Time Vector Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Range", f"[{np.min(time_vec):.6f}, {np.max(time_vec):.6f}]")
    stats_table.add_row("Mean", f"{np.mean(time_vec):.6f}")
    stats_table.add_row("Std Dev", f"{np.std(time_vec):.6f}")

    console.print(stats_table)

    c_indices = pairs[:, 0]  # child indices
    p_indices = pairs[:, 1]  # parent indices
    delta_space_norm = np.linalg.norm(spatial[p_indices] - spatial[c_indices], axis=1)

    for n_pass in range(1, max_passes + 1):
        # Δt = t_parent - t_child
        delta_t = time_vec[p_indices] - time_vec[c_indices]
        # ||Δx|| = ||spatial_parent - spatial_child||

        # Violation: delta_t <= delta_space_norm + eps1
        violation = delta_t <= delta_space_norm + eps1

        if not violation.any():
            # Success panel
            success_panel = Panel.fit(
                f"[bold green]Convergence Successful! 🎉[/bold green]\n"
                f"[green]Passes:[/green] {n_pass}\n"
                f"[green]Final violations:[/green] 0",
                title="✅ Success",
                border_style="green",
            )
            console.print(success_panel)
            return n_pass

        # Check for NaNs or infinite values BEFORE making changes
        if np.isnan(time_vec).any():
            console.print(
                Panel(
                    f"[bold red]NaN detected in time_vec during pass {n_pass}![/bold red]\n"
                    f"[red]NaN indices:[/red] {np.where(np.isnan(time_vec))[0][:10]}...",
                    title="⚠️ NaN Error",
                    border_style="red",
                )
            )
            return max_passes + 2  # Indicate NaN failure

        if np.isinf(time_vec).any():
            console.print(
                Panel(
                    f"[bold red]Infinite values detected in time_vec during pass {n_pass}![/bold red]\n"
                    f"[red]Infinite indices:[/red] {np.where(np.isinf(time_vec))[0][:10]}...",
                    title="⚠️ Infinity Error",
                    border_style="red",
                )
            )
            return max_passes + 3  # Indicate infinity failure

        # Get the child and parent indices for the violating pairs
        c_bad = c_indices[violation]
        p_bad = p_indices[violation]

        shift = delta_space_norm[violation] + eps1 - delta_t[violation] * 0.5
        target_parent_t = time_vec[c_bad] + delta_space_norm[violation] + eps1
        shift_for_parent = target_parent_t - time_vec[p_bad]

        # Debug information for large adjustments
        if n_pass <= 5 or n_pass % 1000 == 0:
            console.print(
                f"[yellow]Pass {n_pass}: {violation.sum()} violations[/yellow]"
            )

        # FIXED: Remove double addition bug - use simpler, more stable update

        # Apply the updates using numpy advanced indexing
        # Increase parent's time
        np.add.at(time_vec, p_bad, shift_for_parent.astype(time_vec.dtype))
        np.add.at(time_vec, p_bad, shift)
        # Decrease child's time
        np.add.at(time_vec, c_bad, -shift)

    # Failure panel
    failure_panel = Panel.fit(
        f"[bold red]Convergence Failed ❌[/bold red]\n"
        f"[red]Max passes reached:[/red] {max_passes}\n"
        f"[red]Remaining violations:[/red] {violation.sum()}",
        title="❌ Convergence Failed",
        border_style="red",
    )
    console.print(failure_panel)
    return max_passes + 1


def project_and_update_embedding(new_event, space, potential_parents, iteration):
    """
    SIMPLIFIED: Adjusts new_event's embedding based on its 'best' parent.
    A real version would be an iterative global adjustment.
    """
    if not potential_parents:  # First event or no clear parent
        new_event.t = float(iteration)  # Keep its iteration time
        # Could re-randomize x_emb, y_emb or keep initial
        return

    # --- Find a "best" parent from recent history ---
    best_parent = None
    sorted_parents = sorted(potential_parents, key=lambda p: p.fitness)
    if sorted_parents:
        best_parent = sorted_parents[0]

    if not best_parent:
        return

    # --- "Causality" Logic ---
    is_improvement = new_event.fitness < best_parent.fitness
    is_significant_improvement = (
        new_event.fitness < best_parent.fitness * 0.9
    )  # e.g. 10% better

    # Rule 1: Significant Improvement (Exploitation) -> Pull into near-null geodesic
    if is_significant_improvement:
        print(
            f"  Significant improvement over parent {best_parent.id}. Pulling closer."
        )
        new_event.t = best_parent.t + 0.1  # Slightly in future of parent
        # Move embedding coords closer to parent's to simulate near-null
        new_event.x_emb = best_parent.x_emb + random.uniform(-0.05, 0.05)
        new_event.y_emb = best_parent.y_emb + random.uniform(-0.05, 0.05)

    # Rule 2: Minor Improvement or Similar -> Maintain some distance
    elif is_improvement:
        print(f"  Minor improvement over parent {best_parent.id}.")
        new_event.t = best_parent.t + 0.5
        # Keep some distance to encourage exploration around this area
        new_event.x_emb = best_parent.x_emb + random.uniform(-0.2, 0.2)
        new_event.y_emb = best_parent.y_emb + random.uniform(-0.2, 0.2)

    # Rule 3: Worse or Exploration -> Push further in time/embedding space
    else:  # Worse solution
        print(f"  Worse than parent {best_parent.id}. Pushing for exploration.")
        new_event.t = float(iteration)  # Keep its own time, or push further from parent
        # Could make it more random in embedding to signify a new exploration path
        new_event.x_emb = random.uniform(-1, 1)
        new_event.y_emb = random.uniform(-1, 1)

    # Ensure t is monotonically increasing with iteration for simplicity here
    new_event.t = max(
        new_event.t, float(iteration) - 0.5
    )  # Allow some pull-back but not too much


# --- Agent (Potential Search Area) ---
class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.solution = [0.0, 0.0]  # Will be set by Runnable
        self.fitness = float("inf")  # Will be set by Runnable
        self.personal_best_solution = [0.0, 0.0]
        self.personal_best_fitness = float("inf")
        self.associated_event_id = None

    def update_personal_best(self):
        if self.fitness < self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_solution = list(self.solution)

    def move(self, global_best_event, space, iteration, objective_function, bounds):
        # --- Physics-Enhanced Movement Logic ---
        w = 0.5  # Inertia
        c1 = 1.0  # Personal best weight
        c2 = 1.5  # Global best weight
        c3 = 0.8  # Minkowski influencer weight
        c4 = 0.3  # Physics energy penalty weight

        current_event = (
            space.get_event(self.associated_event_id)
            if self.associated_event_id
            else None
        )

        # 1. Attraction to personal best
        vec_to_pbest_x = self.personal_best_solution[0] - self.solution[0]
        vec_to_pbest_y = self.personal_best_solution[1] - self.solution[1]

        # 2. Attraction to global best
        vec_to_gbest_x = global_best_event.solution[0] - self.solution[0]
        vec_to_gbest_y = global_best_event.solution[1] - self.solution[1]

        # 3. Physics-aware causal influencer selection
        vec_to_mink_x, vec_to_mink_y = 0, 0
        physics_penalty = 0.0

        if current_event:
            # Use enhanced causal parent selection
            causal_parents = space.get_causal_parents(current_event, max_count=5)

            if causal_parents:
                # Choose the best physics-guided parent
                best_parent = causal_parents[0]  # Already sorted by physics score

                vec_to_mink_x = best_parent.solution[0] - self.solution[0]
                vec_to_mink_y = best_parent.solution[1] - self.solution[1]

                # Calculate physics penalty for current trajectory
                current_pos = [
                    current_event.x_emb,
                    current_event.y_emb,
                    current_event.t,
                ]
                parent_pos = [best_parent.x_emb, best_parent.y_emb, best_parent.t]

                physics_penalty = physics_loss(
                    [(0, 1)], {0: current_pos, 1: parent_pos}
                )

        # 4. Proposed velocity calculation
        vx = (
            w * 0.1 * random.uniform(-1, 1)
            + c1 * random.random() * vec_to_pbest_x
            + c2 * random.random() * vec_to_gbest_x
            + c3 * random.random() * vec_to_mink_x
        )
        vy = (
            w * 0.1 * random.uniform(-1, 1)
            + c1 * random.random() * vec_to_pbest_y
            + c2 * random.random() * vec_to_gbest_y
            + c3 * random.random() * vec_to_mink_y
        )

        # 5. Physics-aware velocity adjustment
        # If physics penalty is high, reduce movement towards causal influencer
        if physics_penalty > 1.0:
            vx -= c4 * physics_penalty * (vec_to_mink_x / (abs(vec_to_mink_x) + 1e-6))
            vy -= c4 * physics_penalty * (vec_to_mink_y / (abs(vec_to_mink_y) + 1e-6))

        # Simple velocity clamping
        max_vel = (bounds[0][1] - bounds[0][0]) * 0.1
        vx = max(-max_vel, min(max_vel, vx))
        vy = max(-max_vel, min(max_vel, vy))

        # 6. Test move and evaluate physics compatibility
        test_solution_x = self.solution[0] + vx
        test_solution_y = self.solution[1] + vy

        # Clamp to bounds
        test_solution_x = max(bounds[0][0], min(test_solution_x, bounds[0][1]))
        test_solution_y = max(bounds[1][0], min(test_solution_y, bounds[1][1]))

        test_fitness = objective_function(test_solution_x, test_solution_y)

        # 7. Accept move if it's beneficial or physics-compatible
        fitness_improvement = self.fitness - test_fitness
        accept_move = fitness_improvement > 0 or (
            fitness_improvement > -0.1 and physics_penalty < 2.0
        )

        if accept_move:
            self.solution[0] = test_solution_x
            self.solution[1] = test_solution_y
            self.fitness = test_fitness
        else:
            # Small random exploration if move is rejected
            self.solution[0] += random.uniform(-0.01, 0.01) * (
                bounds[0][1] - bounds[0][0]
            )
            self.solution[1] += random.uniform(-0.01, 0.01) * (
                bounds[1][1] - bounds[1][0]
            )
            self.solution[0] = max(bounds[0][0], min(self.solution[0], bounds[0][1]))
            self.solution[1] = max(bounds[1][0], min(self.solution[1], bounds[1][1]))
            self.fitness = objective_function(self.solution[0], self.solution[1])

        self.update_personal_best()


# --- Main LMSO Algorithm ---
class Runnable:
    """A configurable runner for the Latent Minkowski Swarm Optimizer"""

    def __init__(self, objective_function, bounds, num_agents=20, num_iterations=50):
        """
        Initialize the optimizer

        Args:
            objective_function: Function to minimize, takes (x, y) and returns float
            bounds: List of tuples [(x_min, x_max), (y_min, y_max)]
            num_agents: Number of agents in the swarm
            num_iterations: Number of optimization iterations
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_agents = num_agents
        self.num_iterations = num_iterations

    def run(self):
        """Run the optimization and return the best result"""
        global causal_predictor

        # Initialize causal predictor
        causal_predictor = CausalConnectionPredictor()

        space = MinkowskiSpace()
        agents = [self._create_agent(i) for i in range(self.num_agents)]

        # Initial embedding of agents
        for i, agent in enumerate(agents):
            event = self._initial_embed_solution(
                agent.solution, agent.fitness, 0, space
            )
            agent.associated_event_id = event.id

        console.print(f"[green]Initial Global Best:[/green] {space.global_best_event}")

        # Track training data for causal learning
        causal_training_data = []
        physics_energy_history = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(
                "[cyan]Optimizing with Physics...", total=self.num_iterations
            )

            for iteration in range(1, self.num_iterations + 1):
                progress.update(main_task, advance=1)

                # Collect training data for causal learning
                if iteration > 10:  # Start collecting after some events exist
                    events = list(space.events.values())
                    for i, e1 in enumerate(events):
                        for j, e2 in enumerate(events[i + 1 :], i + 1):
                            if e2.t > e1.t:  # Ensure time ordering
                                features = extract_edge_features(e1, e2)
                                # Label: 1 if improvement flow, 0 otherwise
                                improvement_label = (
                                    1.0 if e2.fitness < e1.fitness else 0.0
                                )
                                causal_training_data.append(
                                    (features, improvement_label)
                                )

                for agent in agents:
                    if space.global_best_event:
                        agent.move(
                            space.global_best_event,
                            space,
                            iteration,
                            self.objective_function,
                            self.bounds,
                        )

                    new_event = self._initial_embed_solution(
                        agent.solution, agent.fitness, iteration, space
                    )
                    agent.associated_event_id = new_event.id

                    potential_parents = space.get_recent_events(
                        count=self.num_agents * 2
                    )
                    potential_parents = [
                        p for p in potential_parents if p.id != new_event.id
                    ]

                    # Prepare data for project_until_convergence
                    if potential_parents:
                        all_events = [new_event] + potential_parents
                        pairs = np.array(
                            [[0, i + 1] for i in range(len(potential_parents))]
                        )

                        spatial = np.array([[e.x_emb, e.y_emb] for e in all_events])
                        time_vec = np.array([e.t for e in all_events])

                        passes_taken = project_until_convergence(
                            pairs, spatial, time_vec
                        )

                        for i, event in enumerate(all_events):
                            event.t = float(time_vec[i])
                            event.x_emb = float(spatial[i, 0])
                            event.y_emb = float(spatial[i, 1])

                # Physics-based graph evolution every 20 iterations
                if iteration % 20 == 0:
                    # Calculate current total energy
                    events = list(space.events.values())
                    total_energy = 0.0
                    edge_count = 0

                    for i, e1 in enumerate(events):
                        for e2 in events[i + 1 :]:
                            dx = e2.x_emb - e1.x_emb
                            dy = e2.y_emb - e1.y_emb
                            dt = e2.t - e1.t
                            ds2 = -(dt**2) + dx**2 + dy**2

                            if ds2 < -1e-2:  # Causal connection
                                total_energy += np.sqrt(-ds2)
                                edge_count += 1

                    physics_energy_history.append(total_energy)

                    # Apply interaction step for physics evolution
                    interaction_step(space, max_merges=2)

                    # Train causal predictor on accumulated data
                    if len(causal_training_data) > 50:
                        self._train_causal_predictor(
                            causal_training_data[-100:]
                        )  # Use recent data

                if iteration % (self.num_iterations // 10 or 1) == 0:
                    energy = physics_energy_history[-1] if physics_energy_history else 0
                    progress.update(
                        main_task,
                        description=f"[cyan]Iter {iteration}: Best={space.global_best_event.fitness:.6f}, Energy={energy:.2f}",  # noqa: E501
                    )

        console.print(
            "\n[bold green]Physics-Enhanced Optimization Complete![/bold green]"
        )
        console.print(
            f"[green]Best Solution:[/green] {space.global_best_event.solution}"
        )
        console.print(f"[green]Best Fitness:[/green] {space.global_best_event.fitness}")
        console.print(f"[green]Final Events:[/green] {len(space.events)}")
        console.print(
            f"[green]Causal Training Samples:[/green] {len(causal_training_data)}"
        )

        return space.global_best_event, space

    def _create_agent(self, agent_id):
        """Create an agent with random initial position"""
        agent = Agent(agent_id)
        agent.solution = [
            random.uniform(self.bounds[0][0], self.bounds[0][1]),
            random.uniform(self.bounds[1][0], self.bounds[1][1]),
        ]
        agent.fitness = self.objective_function(agent.solution[0], agent.solution[1])
        agent.personal_best_solution = list(agent.solution)
        agent.personal_best_fitness = agent.fitness
        agent.associated_event_id = None
        return agent

    def _initial_embed_solution(self, solution, fitness, iteration, space):
        """Embed a solution in Minkowski space"""
        t_initial = float(iteration)
        x_emb_initial = random.uniform(-1, 1)
        y_emb_initial = random.uniform(-1, 1)
        return space.add_event(
            solution, fitness, t_initial, x_emb_initial, y_emb_initial
        )

    def _train_causal_predictor(self, training_data):
        """Quick training of causal predictor on recent data"""
        global causal_predictor

        if len(training_data) < 10:
            return

        # Prepare training batch
        features = torch.stack([data[0] for data in training_data])
        labels = torch.FloatTensor([data[1] for data in training_data])

        # Quick training (3 epochs)
        optimizer = torch.optim.Adam(causal_predictor.parameters(), lr=0.01)
        causal_predictor.train()

        for _ in range(3):
            optimizer.zero_grad()
            predictions = causal_predictor(features)
            loss = F.binary_cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()

        causal_predictor.eval()


def visualize_embedding_trajectory(events):
    # Convert dict to list if needed
    if isinstance(events, dict):
        events = list(events.values())

    events.sort(key=lambda e: e.t)
    x = [e.x_emb for e in events]
    y = [e.y_emb for e in events]
    fitness = [e.fitness for e in events]
    t = [e.t for e in events]
    colors = plt.cm.viridis(
        (np.array(fitness) - min(fitness)) / (max(fitness) - min(fitness))
    )

    fig, ax = plt.subplots()
    for i in range(len(events)):
        ax.scatter(x[i], y[i], c=[colors[i]], s=30)
    ax.set_title("Embedding Trajectory Colored by Fitness")
    ax.set_xlabel("x_emb")
    ax.set_ylabel("y_emb")
    plt.show()


def visualize_causal_edges(events, space):
    # Convert dict to list if needed
    if isinstance(events, dict):
        events = list(events.values())

    G = nx.DiGraph()
    for e in events:
        G.add_node(e.id, pos=(e.x_emb, e.y_emb), fitness=e.fitness)
    for child in events:
        for parent in space.get_recent_events(count=10):
            if parent.id != child.id and space.is_in_future_light_cone(parent, child):
                improvement = max(0.01, parent.fitness - child.fitness)
                G.add_edge(parent.id, child.id, weight=improvement)

    pos = {e.id: (e.x_emb, e.y_emb) for e in events}
    edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges]
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        node_size=30,
        edge_color="gray",
        width=edge_widths,
        node_color="blue",
        alpha=0.6,
    )
    plt.title("Causal Graph: Edge Thickness = Fitness Improvement")
    plt.show()


def visualize_bottlenecks(events, space):
    # Convert dict to list if needed
    if isinstance(events, dict):
        events = list(events.values())

    G = nx.DiGraph()
    for e in events:
        G.add_node(e.id, pos=(e.x_emb, e.y_emb), fitness=e.fitness)
    for child in events:
        for parent in space.get_recent_events(count=10):
            if parent.id != child.id and space.is_in_future_light_cone(parent, child):
                improvement = max(0.01, parent.fitness - child.fitness)
                G.add_edge(parent.id, child.id, weight=improvement)

    pos = {e.id: (e.x_emb, e.y_emb) for e in events}
    bottlenecks = nx.betweenness_centrality(G)
    top_bottlenecks = sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)[:10]
    node_colors = [
        "red" if node in dict(top_bottlenecks) else "blue" for node in G.nodes()
    ]
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors, node_size=50, edge_color="gray", alpha=0.7)
    plt.title("Causal Bottlenecks (Red Nodes)")
    plt.show()


def visualize_density(events):
    # Convert dict to list if needed
    if isinstance(events, dict):
        events = list(events.values())

    x_coords = [e.x_emb for e in events]
    y_coords = [e.y_emb for e in events]
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=x_coords, y=y_coords, cmap="Reds", fill=True, bw_adjust=0.5)
    plt.title("Event Density in Embedding Space")
    plt.xlabel("x_emb")
    plt.ylabel("y_emb")
    plt.show()


def visualize_clusters(events):
    # Convert dict to list if needed
    if isinstance(events, dict):
        events = list(events.values())

    embeddings = np.array([[e.x_emb, e.y_emb] for e in events])
    clustering = DBSCAN(eps=0.05, min_samples=5).fit(embeddings)
    labels = clustering.labels_
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10", s=50
    )
    plt.title("Clusters of Causally Disconnected Regions")
    plt.colorbar(scatter)
    plt.show()


def plot_physics_energy_map(events, space):
    """Plot Δt vs spatial distance, colored by causal energy."""
    dt_list, dx_list, dy_list, energy_list = [], [], [], []

    for i, e1 in enumerate(events):
        for e2 in events[i + 1 :]:
            if space.is_in_future_light_cone(e1, e2):
                dt = e2.t - e1.t
                dx = e2.x_emb - e1.x_emb
                dy = e2.y_emb - e1.y_emb
                ds2 = -(dt**2) + dx**2 + dy**2
                if ds2 < 0:
                    energy = np.sqrt(-ds2)
                    dt_list.append(dt)
                    dx_list.append(dx)
                    dy_list.append(dy)
                    energy_list.append(energy)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        dt_list,
        np.sqrt(np.array(dx_list) ** 2 + np.array(dy_list) ** 2),
        c=energy_list,
        cmap="plasma",
        alpha=0.7,
    )
    plt.colorbar(sc, label="Causal Energy")
    plt.xlabel("Δt")
    plt.ylabel("Spatial Distance")
    plt.title("Physics Energy Map: Δt vs Spatial Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def curvature_hotspot_analysis(events, space):
    """Detect triangle inequality violations in causal triplets."""
    violations = []
    coords = {e.id: np.array([e.x_emb, e.y_emb]) for e in events}
    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events[i + 1 :], i + 1):
            if not space.is_in_future_light_cone(e1, e2):
                continue
            for k, e3 in enumerate(events[j + 1 :], j + 1):
                if not space.is_in_future_light_cone(e2, e3):
                    continue
                d13 = np.linalg.norm(coords[e1.id] - coords[e3.id])
                d12 = np.linalg.norm(coords[e1.id] - coords[e2.id])
                d23 = np.linalg.norm(coords[e2.id] - coords[e3.id])
                if d13 > d12 + d23 + 0.01:  # Triangle inequality violated
                    violations.append((e1.x_emb, e1.y_emb))

    if violations:
        x, y = zip(*violations)
        plt.figure(figsize=(6, 6))
        sns.kdeplot(x=x, y=y, cmap="coolwarm", fill=True)
        plt.title("Curvature Hotspots (Triangle Inequality Violations)")
        plt.xlabel("x_emb")
        plt.ylabel("y_emb")
        plt.tight_layout()
        plt.show()
    else:
        print("No triangle inequality violations detected.")


def animate_agent_trails(agent_traces, bounds=[(-1, 1), (-1, 1)]):
    """Animate agent positions over time with trails."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    scatters = [ax.plot([], [], "o-", lw=1, markersize=2)[0] for _ in agent_traces]

    def update(frame):
        ax.clear()
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        for i, trace in enumerate(agent_traces):
            if frame < len(trace):
                x, y = zip(*trace[: frame + 1])
                ax.plot(x, y, "o-", lw=1, markersize=2, alpha=0.6)

    ani = FuncAnimation(
        fig, update, frames=max(len(t) for t in agent_traces), interval=200
    )
    plt.close()
    return ani


# --- Example Objective Functions ---


def sphere_function_2d(x, y):
    """Sphere function: f(x,y) = x² + y²
    Global minimum: (0, 0) with value 0
    Recommended bounds: [(-5, 5), (-5, 5)]
    """
    return x**2 + y**2


def rastrigin_function_2d(x, y, A=10):
    """Rastrigin function: multimodal with many local minima
    Global minimum: (0, 0) with value 0
    Recommended bounds: [(-5.12, 5.12), (-5.12, 5.12)]
    """
    return (
        A * 2
        + (x**2 - A * math.cos(2 * math.pi * x))
        + (y**2 - A * math.cos(2 * math.pi * y))
    )


def rosenbrock_function(x, y, a=1, b=100):
    """Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
    Global minimum: (a, a²) = (1, 1) with value 0
    Recommended bounds: [(-2, 2), (-1, 3)]
    """
    return (a - x) ** 2 + b * (y - x**2) ** 2


def ackley_function(x, y):
    """Ackley function: highly multimodal
    Global minimum: (0, 0) with value 0
    Recommended bounds: [(-5, 5), (-5, 5)]
    """
    return (
        -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
        - math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
        + math.e
        + 20
    )


def himmelblau_function(x, y):
    """Himmelblau's function: has four identical local minima
    Global minima: (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    All with value 0. Recommended bounds: [(-5, 5), (-5, 5)]
    """
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def beale_function(x, y):
    """Beale function
    Global minimum: (3, 0.5) with value 0
    Recommended bounds: [(-4.5, 4.5), (-4.5, 4.5)]
    """
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def run_demo(function, visualize, console):
    best, space = None, None
    if function == "rastrigin":
        # Example 1: Rastrigin function (challenging multimodal)
        console.print(
            Panel.fit(
                "[bold cyan]Example 1: Rastrigin Function[/bold cyan]\n"
                "[green]Multimodal function with many local minima[/green]\n"
                "[yellow]Global minimum: (0, 0) → 0[/yellow]",
                title="🎯 Function Test",
                border_style="cyan",
            )
        )

        optimizer1 = Runnable(
            objective_function=rastrigin_function_2d,
            bounds=[(-5.12, 5.12), (-5.12, 5.12)],
            num_agents=20,
            num_iterations=50,
        )
        best, space = optimizer1.run()

        console.print("\n" + "=" * 50 + "\n")
    elif args.function == "rosenbrock":
        # Example 2: Rosenbrock function (valley-shaped)
        """ console.print(Panel.fit(
            "[bold magenta]Example 2: Rosenbrock Function[/bold magenta]\n"
            "[green]Famous 'banana' function with valley shape[/green]\n"
            "[yellow]Global minimum: (1, 1) → 0[/yellow]",
            title="🍌 Function Test",
            border_style="magenta"
        )) """

        optimizer2 = Runnable(
            objective_function=rosenbrock_function,
            bounds=[(-2, 2), (-1, 3)],
            num_agents=25,
            num_iterations=75,
        )
        best, space = optimizer2.run()

        console.print("\n" + "=" * 50 + "\n")
    if args.function == "ackley":
        # Example 3: Ackley function (highly multimodal)
        console.print(
            Panel.fit(
                "[bold green]Example 3: Ackley Function with Physics[/bold green]\n"
                "[green]Highly multimodal with learned causal dynamics[/green]\n"
                "[yellow]Global minimum: (0, 0) → 0[/yellow]",
                title="⚡ Physics Test",
                border_style="green",
            )
        )

        optimizer3 = Runnable(
            objective_function=ackley_function,
            bounds=[(-5, 5), (-5, 5)],
            num_agents=30,
            num_iterations=50,
        )
        best, space = optimizer3.run()
    elif args.function == "himmelblau":
        # Example 4: Himmelblau function (four local minima)
        console.print(
            Panel.fit(
                "[bold magenta]Example 4: Himmelblau Function[/bold magenta]\n"
                "[green]Four local minima with unique solutions[/green]\n"
                "[yellow]Global minimum: (3, 2) → 0[/yellow]",
                title="🌈 Function Test",
                border_style="magenta",
            )
        )
        optimizer4 = Runnable(
            objective_function=himmelblau_function,
            bounds=[(-5, 5), (-5, 5)],
            num_agents=20,
            num_iterations=50,
        )
        best, space = optimizer4.run()
        console.print("\n" + "=" * 50 + "\n")
    if visualize:
        # Visualization with physics annotations
        visualize_embedding_trajectory(space3.events)
        visualize_density(space3.events)

        # === Spacetime Analysis Plugin for LMSO ===
        plot_physics_energy_map(events, space3)
        curvature_hotspot_analysis(events, space3)
        visualize_causal_edges(space3.events, space3)
        visualize_bottlenecks(space3.events, space3)
        visualize_clusters(space3.events)
    return best, space


# --- Demo Usage ---
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--function",
        type=str,
        default="rastrigin",
        help="Select the function to optimize",
        choices=["rastrigin", "rosenbrock", "ackley", "himmelblau", "beale"],
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        help="Visualize the optimization process",
    )
    args = parser.parse_args()

    best, space = run_demo(args.function, args.visualize, console)

    # Physics Analysis
    console.print("\n" + "=" * 60)
    console.print(
        Panel.fit(
            "[bold magenta]🔬 Physics Analysis Results[/bold magenta]",
            title="⚛️ Causal Dynamics",
            border_style="magenta",
        )
    )

    # Analyze final causal structure
    events = list(space.events.values())
    causal_connections = 0
    total_energy = 0.0

    analysis_table = Table(title="📊 Spacetime Physics Summary")
    analysis_table.add_column("Metric", style="cyan", no_wrap=True)
    analysis_table.add_column("Value", style="green")

    for i, e1 in enumerate(events):
        for e2 in events[i + 1 :]:
            if space.is_in_future_light_cone(e1, e2):
                causal_connections += 1
                dx = e2.x_emb - e1.x_emb
                dy = e2.y_emb - e1.y_emb
                dt = e2.t - e1.t
                ds2 = -(dt**2) + dx**2 + dy**2
                if ds2 < 0:
                    total_energy += np.sqrt(-ds2)

    analysis_table.add_row("Final Events", str(len(events)))
    analysis_table.add_row("Causal Connections", str(causal_connections))
    analysis_table.add_row("Total Physics Energy", f"{total_energy:.4f}")
    analysis_table.add_row(
        "Avg Energy/Connection", f"{total_energy / max(causal_connections, 1):.4f}"
    )
    analysis_table.add_row("Solution Quality", f"{best.fitness:.6f}")

    console.print(analysis_table)

    
