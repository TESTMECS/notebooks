import numpy as np
from scipy.linalg import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gymnasium as gym
from collections import deque
class LorentzToSU2:
    """
    Class to convert a Lorentz-like SO(3,1) rotation (pure spatial rotation part)
    into an equivalent SU(2) spinor transformation.
    """

    def __init__(self, rotation_matrix):
        """
        Initialize with a 3x3 rotation matrix (assumed from SO(3) component).
        """
        self.rotation_matrix = np.array(rotation_matrix, dtype=float)
        self.axis, self.angle = self._extract_axis_angle()

    def _extract_axis_angle(self):
        """
        Extract the rotation axis and angle from a 3x3 rotation matrix.
        """
        R = self.rotation_matrix
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))  # Stable arccos

        if np.isclose(angle, 0):
            return np.array([0, 0, 1]), 0  # Default axis for identity rotation

        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        return axis, angle

    def to_su2(self):
        """
        Compute the SU(2) matrix corresponding to the axis-angle rotation.
        """
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        nx, ny, nz = self.axis / norm(self.axis)
        generator = nx * sigma_x + ny * sigma_y + nz * sigma_z
        theta = self.angle
        return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * generator

# Example usage:
# Define a 90-degree rotation around z-axis
theta = np.pi / 2
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1]
])

converter = LorentzToSU2(Rz)
U_su2 = converter.to_su2()
print(U_su2)


class SU2Attention(nn.Module):
    """
    SU(2)-inspired attention layer using 2D complex spinor representations.
    Each token is represented by a complex 2-vector (spinor).
    Attention weights are computed via Hermitian inner product (phase-aware).
    """

    def __init__(self, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.scale = 1.0 / (2 ** 0.5)

    def forward(self, spinors):
        """
        spinors: (batch, seq_len, heads, 2) complex-valued tensor
        returns: (batch, seq_len, heads, 2) transformed spinors
        """
        batch, seq_len, heads, _ = spinors.shape
        assert heads == self.n_heads, "Head mismatch"

        # Normalize spinors to unit length
        normed = spinors / torch.norm(spinors, dim=-1, keepdim=True)

        # Compute attention weights via Hermitian inner product
        # (spinor_i^† @ spinor_j) for all i, j in seq
        spinor_conj = normed.conj().unsqueeze(2)  # (B, T, 1, H, 2)
        spinor_base = normed.unsqueeze(1)         # (B, 1, T, H, 2)
        attn_logits = torch.einsum("bijhd,bijhd->bijh", spinor_conj, spinor_base)  # (B, T, T, H)
        attn_weights = F.softmax(attn_logits.real * self.scale, dim=2)  # (B, T, T, H)

        # Apply attention: weighted sum of spinors
        # Convert attn_weights to complex to match spinors type
        attn_weights_complex = attn_weights.to(dtype=spinors.dtype)
        out = torch.einsum("bijh,bjhd->bihd", attn_weights_complex, spinors)  # (B, T, H, 2)
        return out

# Example: 4 tokens, batch=1, heads=2
batch, seq_len, heads = 1, 4, 2
spinors = torch.randn(batch, seq_len, heads, 2, dtype=torch.cfloat)

attn = SU2Attention(n_heads=heads)
out = attn(spinors)
print(out)

class SU2Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.axis = nn.Parameter(torch.randn(3))
        self.angle = nn.Parameter(torch.tensor(0.0))

    def forward(self, spinors):
        axis = self.axis / torch.norm(self.axis)
        theta = self.angle
        nx, ny, nz = axis

        # Pauli matrices
        σx = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
        σy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat)
        σz = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
        I = torch.eye(2, dtype=torch.cfloat)

        G = torch.cos(theta/2)*I - 1j*torch.sin(theta/2)*(nx*σx + ny*σy + nz*σz)
        return spinors @ G


class QLearningAgent(nn.Module):
    """
    SU(2)-based Q-Learning Agent.
    Uses a set of learnable SU(2) gates as discrete actions.
    Maps CartPole observations to SU(2) spinor space.
    """
    def __init__(self, obs_dim=4, n_actions=2, lr=0.001, gamma=0.99):
        super().__init__()
        self.actions = nn.ModuleList([SU2Gate() for _ in range(n_actions)])
        
        # Neural network to map observations to Q-values
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )
        
        # Neural network to map observations to spinor state
        self.obs_to_spinor = nn.Sequential(
            nn.Linear(obs_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4)  # 4 real values -> 2 complex values
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions
        self.eps = 1.0
        self.eps_decay = 0.9995  # Slower decay
        self.eps_min = 0.01  # Lower minimum exploration for less noise
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64  # Larger batch for more stable updates
        
        # Target network for stability
        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )
        self.target_encoder.load_state_dict(self.obs_encoder.state_dict())
        self.target_update_freq = 100  # Update target every 100 steps
        self.step_count = 0

    def obs_to_su2_state(self, obs):
        """Convert CartPole observation to SU(2) spinor state"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        encoded = self.obs_to_spinor(obs_tensor)
        # Convert 4 real values to 2 complex values
        real_part = encoded[:2]
        imag_part = encoded[2:]
        spinor = torch.complex(real_part, imag_part)
        # Normalize to unit spinor
        return spinor / torch.norm(spinor)

    def select_action(self, obs):
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        q_values = self.obs_encoder(obs_tensor)
        return torch.argmax(q_values).item()

    def apply_action(self, action_idx, state):
        gate = self.actions[action_idx]
        return gate(state)

    def remember(self, obs, action, reward, next_obs, done):
        """Store experience in replay buffer"""
        self.memory.append((obs, action, reward, next_obs, done))
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        batch = random.sample(self.memory, self.batch_size)
        obs_batch = torch.tensor([e[0] for e in batch], dtype=torch.float32)
        action_batch = torch.tensor([e[1] for e in batch], dtype=torch.long)
        reward_batch = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_obs_batch = torch.tensor([e[3] for e in batch], dtype=torch.float32)
        done_batch = torch.tensor([e[4] for e in batch], dtype=torch.bool)
        
        current_q_values = self.obs_encoder(obs_batch).gather(1, action_batch.unsqueeze(1))
        # Use target network for more stable targets
        next_q_values = self.target_encoder(next_obs_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_encoder.load_state_dict(self.obs_encoder.state_dict())
        
        # Decay epsilon
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        
        return loss.item()

    def update(self, obs, action, reward, next_obs, done):
        """Simple update method for immediate learning"""
        self.remember(obs, action, reward, next_obs, done)
        return self.replay()
# Initialize agent and example spinor state
agent = QLearningAgent()
initial_spinor = torch.tensor([1, 0], dtype=torch.cfloat)

# Environment loop
env = gym.make("CartPole-v1")
agent = QLearningAgent(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n)

for episode in range(300):  # More episodes for learning
    obs, _ = env.reset()
    total_reward = 0

    for t in range(500):  # CartPole max steps
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Convert observations to SU(2) states for demonstration
        state = agent.obs_to_su2_state(obs)
        next_state = agent.obs_to_su2_state(next_obs)
        
        # Apply SU(2) transformation (for research purposes)
        transformed_state = agent.apply_action(action, state)
        
        # Update Q-learning with proper state transitions
        agent.update(obs, action, reward, next_obs, done)
        
        obs = next_obs
        total_reward += reward
        if done:
            break

    if episode % 10 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.1f}, Epsilon = {agent.eps:.3f}")