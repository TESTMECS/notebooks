# %%
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt


# %%
class OptAEGV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vx = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.vy = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.wx = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.wy = nn.Parameter(th.rand(1, 1, 1, dtype=th.complex64) / 100)
        self.afactor = nn.Parameter(th.rand(1, 1, dtype=th.complex64) / 100)
        self.mfactor = nn.Parameter(th.rand(1, 1, dtype=th.complex64) / 100)

    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    def normalize(self, z):
        rho = th.abs(z)
        theta = th.atan2(th.imag(z), th.real(z))
        is_zero = rho < 1e-9
        # For phase, if magnitude is zero, default to e^(i*0) = 1.
        # Otherwise, use z/rho for numerical stability if rho is not tiny, or cos+isin.
        # Using cos+isin is generally safer for phase preservation.
        normalized_phase = th.cos(theta) + 1.0j * th.sin(theta)
        safe_phase = th.where(is_zero, th.ones_like(normalized_phase), normalized_phase)
        return th.tanh(rho) * safe_phase

    def forward(self, data):
        shape = data.size()
        data_flat = data.flatten(1)
        data_norm = self.normalize(data_flat)
        b = shape[0]
        data_view = data_norm.view(b, -1, 1)
        v = self.flow(self.vx, self.vy, data_view)
        w = self.flow(self.wx, self.wy, data_view)
        dx = self.afactor * (v * th.tanh(w)).squeeze(-1)
        dy = self.mfactor * th.tanh(data_norm)
        data_flow = self.flow(dx, dy, data_norm)
        data_final = self.normalize(data_flow)
        return data_final.view(*shape)


# %%
class SpatiallyMaskedSelfAttention_Light(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        patch_coords_H,
        patch_coords_W,
        radius_squared,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        coords = []
        for i in range(patch_coords_H):
            for j in range(patch_coords_W):
                coords.append((i, j))
        # Register as buffer so it moves with the model to device
        self.register_buffer("patch_coords", th.tensor(coords, dtype=th.float32))

        self.radius_squared = radius_squared

        _N_patches = self.patch_coords.shape[0]
        coords_q = self.patch_coords.unsqueeze(0)
        coords_k = self.patch_coords.unsqueeze(1)
        distance_squared_matrix = th.sum((coords_q - coords_k) ** 2, dim=-1)
        self.register_buffer(
            "spatial_mask",
            (distance_squared_matrix > self.radius_squared),
        )

    def forward(self, x):
        B, N, C = x.shape  # N = H*W, C = dim_features
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        # Mask is (N,N), unsqueeze for batch and heads
        attn_scores = attn_scores.masked_fill(
            self.spatial_mask.unsqueeze(0).unsqueeze(0), -1e9
        )
        attn_probs = F.softmax(attn_scores, dim=-1)

        output = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        return output


# %%
class RG_AEG_Block_SpatialAttn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        H_in=28,
        W_in=28,
        use_attention=False,
        attn_heads=2,
        attn_radius_squared=4.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=4,
            stride=stride,
            padding=1,
        )
        self.aeg = OptAEGV3()
        self.out_channels = out_channels  # complex out channels
        self.use_attention = use_attention

        self.H_after_conv = (H_in - 4 + 2 * 1) // stride + 1
        self.W_after_conv = (W_in - 4 + 2 * 1) // stride + 1
        self.attention_dim = out_channels * 2

        if self.use_attention:
            self.norm_after_conv = nn.LayerNorm(
                [
                    self.attention_dim,
                    self.H_after_conv,
                    self.W_after_conv,
                ]
            )
            self.spatial_attn = SpatiallyMaskedSelfAttention_Light(
                dim=self.attention_dim,
                num_heads=attn_heads,
                patch_coords_H=self.H_after_conv,
                patch_coords_W=self.W_after_conv,
                radius_squared=attn_radius_squared,
            )
        # Norms before AEG apply whether attention is used or not on the (potentially attended) features
        self.norm_before_aeg_real = nn.LayerNorm(
            [
                out_channels,
                self.H_after_conv,
                self.W_after_conv,
            ]
        )
        self.norm_before_aeg_imag = nn.LayerNorm(
            [
                out_channels,
                self.H_after_conv,
                self.W_after_conv,
            ]
        )

    def forward(self, x):
        if x.is_complex():
            x = th.cat([x.real, x.imag], dim=1)

        x_conv = self.conv(x)  # [B, out_channels*2, H_out, W_out]

        if self.use_attention:
            x_normed_conv = self.norm_after_conv(x_conv)
            B, C_att, H_att, W_att = x_normed_conv.shape
            x_seq = x_normed_conv.flatten(2).transpose(1, 2)  # [B, H_att*W_att, C_att]
            x_attended = self.spatial_attn(x_seq)
            x_conv_attended = x_attended.transpose(1, 2).reshape(B, C_att, H_att, W_att)
            x_for_aeg_real_imag = x_conv + x_conv_attended  # Residual connection
        else:
            x_for_aeg_real_imag = x_conv

        x_real_part = self.norm_before_aeg_real(
            x_for_aeg_real_imag[:, : self.out_channels]
        )
        x_imag_part = self.norm_before_aeg_imag(
            x_for_aeg_real_imag[:, self.out_channels :]
        )
        x_complex = x_real_part + 1.0j * x_imag_part

        x_aeg = self.aeg(x_complex)
        x_disentangled = th.cat([x_aeg.real, x_aeg.imag], dim=1)
        return x_aeg, x_disentangled


# %%
class RG_AEG_Network_SpatialAttn(nn.Module):
    def __init__(
        self,
        attention_in_block2=False,
        attn_radius_sq_b2=2.0,
        attn_heads_b2=2,
    ):
        super().__init__()
        self.block1 = RG_AEG_Block_SpatialAttn(
            in_channels=1,
            out_channels=4,
            H_in=28,
            W_in=28,
            use_attention=False,
        )
        self.block2 = RG_AEG_Block_SpatialAttn(
            in_channels=8,
            out_channels=8,
            H_in=14,
            W_in=14,
            use_attention=attention_in_block2,
            attn_heads=attn_heads_b2,
            attn_radius_squared=attn_radius_sq_b2,
        )
        self.block3 = RG_AEG_Block_SpatialAttn(
            in_channels=16,
            out_channels=8,
            H_in=7,
            W_in=7,
            use_attention=False,
        )
        self.fc = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        _, x_r1 = self.block1(x)
        _, x_r2 = self.block2(x_r1)
        _, x_r3 = self.block3(x_r2)
        x_flat = th.flatten(x_r3, 1)
        output = self.fc(x_flat)
        return output


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_rg_aeg(model, model_name="Model"):
    param_count = count_parameters(model)
    print(f"\n--- {model_name} Analysis ---")
    print(f"Total parameters: {param_count}")

    block_params = {"block1": 0, "block2": 0, "block3": 0, "fc": 0}
    for (
        name,
        module,
    ) in model.named_children():  # Iterate over direct children
        block_param_count = count_parameters(module)
        if name.startswith("block"):
            block_params[name] = block_param_count
        elif name.startswith("fc"):
            block_params["fc"] = block_param_count

    print("Parameters per main component:")
    for block_name, params in block_params.items():
        print(f"  {block_name}: {params}")
    return param_count


# %%
def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch_num,
    num_epochs,
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)  # Sum of losses
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        if batch_idx % 200 == 0:
            print(
                f"  Epoch {epoch_num}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}"
            )
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with th.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # Sum of losses
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# %%
def run_experiment(model_type="vanilla", attention_radius_squared=2.0, num_epochs=15):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(
        f"\n=== Running Experiment: {model_type.upper()} RG-AEG (RadiusSq={attention_radius_squared if model_type == 'spatial_attn' else 'N/A'}) ==="
    )
    print(f"Using device: {device}")

    transform_mnist = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    if not os.path.exists("./data"):
        os.makedirs("./data")
    train_dataset_mnist = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform_mnist
    )
    test_dataset_mnist = torchvision.datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transform_mnist,
    )
    train_loader_mnist = DataLoader(
        train_dataset_mnist,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )
    test_loader_mnist = DataLoader(
        test_dataset_mnist,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    if model_type == "spatial_attn":
        model = RG_AEG_Network_SpatialAttn(
            attention_in_block2=True,
            attn_radius_sq_b2=attention_radius_squared,
            attn_heads_b2=2,
        ).to(device)
        model_name_str = f"RG-AEG SpatialAttn Rsq{attention_radius_squared}"
    else:  # Vanilla RG-AEG
        model = RG_AEG_Network_SpatialAttn(attention_in_block2=False).to(
            device
        )  # attention_in_block2=False makes it vanilla
        model_name_str = "RG-AEG Vanilla"

    num_params = analyze_model_rg_aeg(model, model_name_str)

    optimizer = th.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_test_acc = 0

    print(f"Starting Training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model,
            train_loader_mnist,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
        )
        test_loss, test_acc = evaluate(model, test_loader_mnist, criterion, device)
        scheduler.step()
        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"*** New best test accuracy: {best_test_acc:.2f}% ***")

        print(
            f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.2f}s"
        )

    print(
        f"Training finished for {model_name_str}. Best test accuracy: {best_test_acc:.2f}%"
    )
    return model_name_str, history, num_params, best_test_acc


# %%
NUM_EPOCHS_MAIN = 15  # As in the original notebook, can be reduced for faster runs
results_all_models = {}

# Run Vanilla RG-AEG
(
    name_vanilla,
    history_vanilla,
    params_vanilla,
    best_acc_vanilla,
) = run_experiment(model_type="vanilla", num_epochs=NUM_EPOCHS_MAIN)
results_all_models[name_vanilla] = {
    "history": history_vanilla,
    "params": params_vanilla,
    "best_acc": best_acc_vanilla,
}

# Run Spatially Masked RG-AEG (Attention in Block 2)
# Radius sqrt(2) for 7x7 grid (block 2 output H,W) means squared radius = 2.0.
# This allows attention to immediate diagonal neighbors.
radius_sq_b2 = 2.0
(
    name_spatial,
    history_spatial,
    params_spatial,
    best_acc_spatial,
) = run_experiment(
    model_type="spatial_attn",
    attention_radius_squared=radius_sq_b2,
    num_epochs=NUM_EPOCHS_MAIN,
)
results_all_models[name_spatial] = {
    "history": history_spatial,
    "params": params_spatial,
    "best_acc": best_acc_spatial,
}

# --- Plotting Combined Training History ---
plt.figure(figsize=(14, 5))
epochs_range = range(1, NUM_EPOCHS_MAIN + 1)

plt.subplot(1, 2, 1)
for name, res in results_all_models.items():
    plt.plot(
        epochs_range,
        res["history"]["test_loss"],
        label=f"{name} (Test Loss)",
    )
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss Comparison")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for name, res in results_all_models.items():
    plt.plot(
        epochs_range,
        res["history"]["test_acc"],
        label=f"{name} (Test Acc)",
    )
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("rg_aeg_spatial_attn_vs_vanilla_history.png", dpi=300)
plt.show()

print("\n--- Final Summary ---")
for name, res in results_all_models.items():
    print(
        f"Model: {name}, Parameters: {res['params']}, Best Test Accuracy: {res['best_acc']:.2f}%"
    )
# --- Final Summary ---
# Model: RG-AEG Vanilla, Parameters: 12772, Best Test Accuracy: 98.75%
# Model: RG-AEG SpatialAttn Rsq2.0, Parameters: 15364, Best Test Accuracy: 98.78% Should be able to run >98.6% with just 12K params!
