import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import numpy as np
import gymnasium as gym
from tqdm import trange

class HamiltonianNet(nn.Module):
    """Learns scalar Hamiltonian H(q, p) from input state."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)  # scalar output


class HamiltonianDynamics(nn.Module):
    """Computes dq/dt and dp/dt using Hamiltonian mechanics."""
    def __init__(self, hamiltonian_fn, input_dim):
        super().__init__()
        self.hamiltonian_fn = hamiltonian_fn
        self.input_dim = input_dim

    def forward(self, state):
        state.requires_grad_(True)
        H = self.hamiltonian_fn(state)
        grad = autograd.grad(H.sum(), state, create_graph=True)[0]
        dq_dt, dp_dt = grad.chunk(2, dim=-1)
        return torch.cat([dp_dt, -dq_dt], dim=-1)


class InvertibleCoupling(nn.Module):
    """Simple additive invertible layer."""
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim // 2, dim // 2), nn.Tanh(), nn.Linear(dim // 2, dim // 2)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=-1)
        if not reverse:
            x2 = x2 + self.f(x1)
        else:
            x2 = x2 - self.f(x1)
        return torch.cat([x1, x2], dim=-1)


class HamiltonianINN(nn.Module):
    """Full model: optional invertible preprocessing + HNN dynamics."""
    def __init__(self, input_dim):
        super().__init__()
        self.coupling = InvertibleCoupling(input_dim)
        self.hamiltonian = HamiltonianNet(input_dim)
        self.dynamics = HamiltonianDynamics(self.hamiltonian, input_dim)

    def forward(self, state, reverse=False):
        x = self.coupling(state, reverse=reverse)
        dx_dt = self.dynamics(x)
        return dx_dt


class ForcingNet(nn.Module):
    """Learns additional non-Hamiltonian force/dissipation."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class GeneralizedHamiltonianDynamics(nn.Module):
    """Full dynamics: J∇H + F"""

    def __init__(self, H_net: HamiltonianNet, F_net: ForcingNet):
        super().__init__()
        self.H = H_net
        self.F = F_net

    def forward(self, z):
        """z = [q, p] (or more general state)"""
        z.requires_grad_(True)
        H = self.H(z)
        gradH = autograd.grad(H.sum(), z, create_graph=True)[0]

        # Canonical symplectic structure
        dim = z.shape[1] // 2
        dq_dt = gradH[:, dim:]  # ∂H/∂p
        dp_dt = -gradH[:, :dim]  # -∂H/∂q

        hnn_term = torch.cat([dq_dt, dp_dt], dim=-1)
        forcing_term = self.F(z)

        return hnn_term + forcing_term


class SpringDynamics:
    """Spring oscillator dynamics and Hamiltonian Neural Network training."""
    
    @staticmethod
    def spring_dynamics(t, state, k=1.0):
        """True dynamics for mass-spring oscillator."""
        q, p = state[..., 0], state[..., 1]
        dqdt = p
        dpdt = -k * q
        return torch.stack([dqdt, dpdt], dim=-1)

    @staticmethod
    def generate_trajectory(timesteps=100, dt=0.1):
        """Generate synthetic trajectory data."""
        t = torch.linspace(0, timesteps * dt, timesteps)
        q0, p0 = torch.tensor([1.0]), torch.tensor([0.0])
        state0 = torch.stack([q0, p0], dim=-1)
        traj = odeint(
            SpringDynamics.spring_dynamics, state0, t, method="dopri5"
        ).squeeze(1)
        return t, traj

    class HNN(nn.Module):
        """Hamiltonian Neural Network for spring system."""
        def __init__(self, input_dim=2):
            super().__init__()
            self.hamiltonian = HamiltonianNet(input_dim)
            self.dynamics = HamiltonianDynamics(self.hamiltonian, input_dim)

        def forward(self, t_or_state, state=None):
            # Handle both (t, state) and (state) calling conventions
            if state is None:
                # Called with just state (during training)
                return self.dynamics(t_or_state)
            else:
                # Called with (t, state) from odeint
                return self.dynamics(state)

    @staticmethod
    def train(model, trajectory, t, epochs=2000, lr=1e-3):
        """Train the Hamiltonian Neural Network."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            state = trajectory[:-1]
            target = (trajectory[1:] - trajectory[:-1]) / (t[1] - t[0])
            pred = model(state)
            loss = ((pred - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss {loss.item():.6f}")

    @staticmethod
    def plot_trajectory(true_traj, pred_traj=None):
        """Visualize phase space trajectory."""
        plt.figure(figsize=(6, 4))
        plt.plot(true_traj[:, 0], true_traj[:, 1], label="True", lw=2)
        if pred_traj is not None:
            plt.plot(pred_traj[:, 0], pred_traj[:, 1], "--", label="HNN Predicted")
        plt.xlabel("q (position)")
        plt.ylabel("p (momentum)")
        plt.legend()
        plt.title("Phase Space")
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_energy(traj, model):
        """Plot energy conservation."""
        H_vals = model.hamiltonian(traj).detach().numpy()
        plt.figure(figsize=(6, 4))
        plt.plot(H_vals)
        plt.title("Energy Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Hamiltonian")
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_spring():
        """Main function to run spring oscillator test."""
        print("Running Spring Oscillator Test...")
        t, traj = SpringDynamics.generate_trajectory()
        model = SpringDynamics.HNN(input_dim=2)
        SpringDynamics.train(model, traj, t)

        # Predict future trajectory from initial state
        pred_traj = odeint(model, traj[0], t).squeeze(1).detach()

        # Plotting
        SpringDynamics.plot_trajectory(traj, pred_traj)
        SpringDynamics.plot_energy(pred_traj, model)
        print("Spring test completed!\n")


class CartPoleTest:
    """CartPole dynamics learning with Hamiltonian Neural Network."""
    
    @staticmethod
    def collect_data(env, episodes=100):
        """Collect trajectory data from CartPole environment."""
        state_list, delta_list = [], []
        for _ in trange(episodes, desc="Collecting CartPole data"):
            s = env.reset()[0]
            done = False
            while not done:
                a = env.action_space.sample()  # random actions
                s1, _, done, _, _ = env.step(a)
                state_list.append(s)
                delta_list.append(
                    (np.array(s1) - np.array(s)) / 0.02
                )  # approx. dt = 0.02
                s = s1
        states = torch.tensor(np.array(state_list), dtype=torch.float32)
        deltas = torch.tensor(np.array(delta_list), dtype=torch.float32)
        return states, deltas

    @staticmethod
    def train(model, states, targets, epochs=1000, lr=1e-3):
        """Train the Hamiltonian dynamics model."""
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            pred = model(states)
            loss = ((pred - targets) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    @staticmethod
    def run_cartpole(use_forcing_net=False):
        """Main function to run CartPole test."""
        dynamics_type = "Generalized HNN" if use_forcing_net else "Standard HNN"
        print(f"Running CartPole Test with {dynamics_type}...")
        
        env = gym.make("CartPole-v1", render_mode=None)
        states, derivs = CartPoleTest.collect_data(env, episodes=300)

        if use_forcing_net:
            # Use Generalized Hamiltonian Dynamics with forcing term
            hnn = HamiltonianNet(input_dim=4)
            forcing_net = ForcingNet(input_dim=4)
            dyn = GeneralizedHamiltonianDynamics(hnn, forcing_net)
        else:
            # Use standard Hamiltonian Dynamics
            hnn = HamiltonianNet(input_dim=4)
            dyn = HamiltonianDynamics(hnn, input_dim=4)
        
        CartPoleTest.train(dyn, states, derivs)

        # Visualize a few predictions
        pred = dyn(states[:200].requires_grad_()).detach()
        
        plt.figure(figsize=(12, 5))
        plt.suptitle(f"CartPole Dynamics Learning - {dynamics_type}", fontsize=14)
        
        plt.subplot(1, 3, 1)
        plt.plot(pred[:, 2], label="Predicted θ̇")
        plt.plot(derivs[:200, 2], label="True θ̇", alpha=0.6)
        plt.legend()
        plt.title("Angular Velocity")
        plt.xlabel("Step")
        plt.ylabel("θ̇")
        plt.grid()
        
        plt.subplot(1, 3, 2)
        plt.plot(pred[:, 0], label="Predicted ẋ")
        plt.plot(derivs[:200, 0], label="True ẋ", alpha=0.6)
        plt.legend()
        plt.title("Cart Velocity")
        plt.xlabel("Step")
        plt.ylabel("ẋ")
        plt.grid()
        
        plt.subplot(1, 3, 3)
        # Plot prediction error
        error_theta = torch.abs(pred[:, 2] - derivs[:200, 2])
        error_x = torch.abs(pred[:, 0] - derivs[:200, 0])
        plt.plot(error_theta, label="θ̇ Error", alpha=0.7)
        plt.plot(error_x, label="ẋ Error", alpha=0.7)
        plt.legend()
        plt.title("Prediction Errors")
        plt.xlabel("Step")
        plt.ylabel("Absolute Error")
        plt.grid()
        
        plt.tight_layout()
        plt.show()
        env.close()
        
        # Calculate and return performance metrics
        mse_theta = torch.mean((pred[:, 2] - derivs[:200, 2]) ** 2).item()
        mse_x = torch.mean((pred[:, 0] - derivs[:200, 0]) ** 2).item()
        
        print(f"{dynamics_type} Results:")
        print(f"  MSE θ̇: {mse_theta:.6f}")
        print(f"  MSE ẋ: {mse_x:.6f}")
        print(f"  Total MSE: {mse_theta + mse_x:.6f}")
        print("CartPole test completed!\n")
        
        return mse_theta + mse_x  # Return total MSE for comparison


def compare_cartpole_approaches():
    """Compare standard vs generalized Hamiltonian dynamics on CartPole."""
    print("=== CartPole Approach Comparison ===\n")
    
    # Standard Hamiltonian Neural Network
    standard_mse = CartPoleTest.run_cartpole(use_forcing_net=False)
    
    # Generalized Hamiltonian Neural Network with Forcing
    generalized_mse = CartPoleTest.run_cartpole(use_forcing_net=True)
    
    # Compare results
    print("=== Final Comparison ===")
    print(f"Standard HNN Total MSE: {standard_mse:.6f}")
    print(f"Generalized HNN Total MSE: {generalized_mse:.6f}")
    
    if generalized_mse < standard_mse:
        improvement = ((standard_mse - generalized_mse) / standard_mse) * 100
        print(f"✅ Generalized HNN performs {improvement:.1f}% better!")
    else:
        degradation = ((generalized_mse - standard_mse) / standard_mse) * 100
        print(f"❌ Generalized HNN performs {degradation:.1f}% worse.")
    
    return standard_mse, generalized_mse


def main():
    """Run both tests."""
    print("=== Hamiltonian Neural Network Tests ===\n")
    
    # Run Spring Oscillator Test
    SpringDynamics.run_spring()
    
    # Run CartPole comparison
    compare_cartpole_approaches()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 