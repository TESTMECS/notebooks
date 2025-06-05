# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "torchvision",
#     "numpy",
#     "faiss-cpu",
#     "argparse",
#     "pathlib",
#
# ]
# ///
#!/usr/bin/env python
# rg_aeg_retroinfer_demo.py
# ---------------------------------------------------------------
# A compact demo that combines RG‑AEG blocks with a RetroInfer‑
# inspired sparse‑retrieval layer to reduce GPU‑memory pressure.
# Includes optional global attention tokens for sparse retrieval.
#
# Tested with PyTorch 2.2+, CUDA 11.x, FAISS‑CPU 1.7+.
# ---------------------------------------------------------------
import argparse, os, time, math, sys, warnings
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------------------------------------------
# -------- 1. Optional dependency install (for Colab convenience) ---
# ------------------------------------------------------------------
try:
    import faiss  # type: ignore
except ImportError:
    print("Installing FAISS‑CPU …")
    os.system("pip -q install faiss-cpu")  # noqa: S605,S607
    import faiss  # type: ignore


# ===== Normality‑Normalization helpers ======================================
def _safe_clamp(x, min_val=1e-8, max_val=1e8):
    return th.clamp(
        th.nan_to_num(x, nan=0.0, posinf=max_val, neginf=-max_val),
        min=min_val,
        max=max_val,
    )


def estimate_lambda_hat(h: th.Tensor, dims):
    var = h.abs().var(dim=dims, keepdim=True, unbiased=False)
    lambda_hat = 1.0 / (_safe_clamp(1.0 + var))
    return lambda_hat


def psi(h: th.Tensor, lam: th.Tensor):
    eps = 1e-6
    lam = lam.expand_as(h)
    abs_h = _safe_clamp(h.abs(), eps)
    power_part = th.sign(h) * (abs_h**lam)
    log_part = th.sign(h) * th.log1p(abs_h)
    use_log = (lam.abs() < 1e-3).float()
    return use_log * log_part + (1.0 - use_log) * power_part


class NormalityNormalization(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        noise_factor=0.1,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.noise_factor = noise_factor
        if len(self.normalized_shape) == 1:
            self.num_features = self.normalized_shape[0]
        else:
            self.num_features = None
        if self.elementwise_affine:
            param_shape = (
                (self.num_features,)
                if self.num_features is not None
                else self.normalized_shape
            )
            self.gamma = nn.Parameter(th.ones(param_shape))
            self.beta = nn.Parameter(th.zeros(param_shape))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x):
        input_shape = x.shape
        input_ndim = x.ndim
        if th.isnan(x).any() or th.isinf(x).any():
            warnings.warn("NaN/Inf detected in input 'x'. Clipping.")
            x = th.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if input_ndim == 4 and len(self.normalized_shape) == 1:
            assert self.normalized_shape[0] == input_shape[1], (
                f"Expected {self.normalized_shape[0]} channels, got {input_shape[1]}"
            )
            dims = (2, 3)
            N = input_shape[2] * input_shape[3]
            affine_shape = (1, self.num_features, 1, 1)
        elif (
            input_ndim >= 2
            and self.normalized_shape
            == input_shape[-len(self.normalized_shape) :]
        ):
            norm_shape_len = len(self.normalized_shape)
            dims = tuple(
                range(input_ndim - norm_shape_len, input_ndim)
            )
            N = math.prod(self.normalized_shape)
            affine_shape = [1] * (input_ndim - norm_shape_len) + list(
                self.normalized_shape
            )
        else:
            raise ValueError(
                f"Input shape {input_shape} and normalized_shape {self.normalized_shape} are incompatible."
            )

        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        var = th.clamp(var, min=self.eps * self.eps)
        std = th.sqrt(var)
        h = (x - mean) / std

        if th.isnan(h).any() or th.isinf(h).any():
            warnings.warn(
                "NaN/Inf detected after standardization 'h'. Clipping."
            )
            h = th.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        lambda_hat = estimate_lambda_hat(h.detach(), dims=dims)
        x_transformed = psi(h, lambda_hat)
        y = x_transformed
        if self.training and self.noise_factor > 0.0:
            with th.no_grad():
                if th.isnan(y).any() or th.isinf(y).any():
                    warnings.warn(
                        "NaN/Inf detected before noise. Clipping."
                    )
                    y = th.nan_to_num(
                        y, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                xt_mean = y.mean(dim=dims, keepdim=True)
                if N == 0:
                    N = 1
                s = (
                    th.linalg.norm(
                        y - xt_mean, ord=1, dim=dims, keepdim=True
                    )
                    / N
                )
                s = th.nan_to_num(s, nan=0.0)
                s = th.clamp(s, min=0.0, max=1e6)
            noise = th.randn_like(y)
            y = y + noise * self.noise_factor * s
        if th.isnan(y).any() or th.isinf(y).any():
            warnings.warn("NaN/Inf detected before affine. Clipping.")
            y = th.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.elementwise_affine:
            gamma_reshaped = self.gamma.view(affine_shape)
            beta_reshaped = self.beta.view(affine_shape)
            out = y * gamma_reshaped + beta_reshaped
        else:
            out = y
        if th.isnan(out).any() or th.isinf(out).any():
            warnings.warn(
                "NaN/Inf detected in final output. Clipping."
            )
            out = th.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return out

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, noise_factor={noise_factor}".format(
            normalized_shape=self.normalized_shape,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
            noise_factor=self.noise_factor,
        )


# ------------------------------------------------------------------
# 2. RetroInfer‑style sparse retrieval helper
# ------------------------------------------------------------------
class SparseRetriever(nn.Module):
    def __init__(
        self,
        top_k: int = 32,
        init_alpha: float = 0.2,
        learnable_alpha: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.alpha = (
            nn.Parameter(th.tensor(init_alpha))
            if learnable_alpha
            else init_alpha
        )

    @staticmethod
    def _to_np(x: th.Tensor) -> np.ndarray:
        return x.detach().cpu().contiguous().float().numpy()

    def forward(
        self, x_cmp: th.Tensor, attention_tokens: th.Tensor = None
    ) -> th.Tensor:
        if not th.is_complex(x_cmp):
            raise RuntimeError(
                "SparseRetriever expects complex input x_cmp"
            )
        if attention_tokens is not None and not th.is_complex(
            attention_tokens
        ):
            raise RuntimeError(
                "SparseRetriever expects complex input attention_tokens"
            )

        b_query, f = x_cmp.shape
        if b_query == 0:  # No query items
            return x_cmp

        # --- 1. Prepare database for FAISS index ---
        # It will contain current batch features and optionally global attention tokens.
        db_complex_tensors = [x_cmp]
        db_reim_tensors_for_faiss = [
            th.cat([x_cmp.real, x_cmp.imag], dim=1)
        ]

        if (
            attention_tokens is not None
            and attention_tokens.shape[0] > 0
        ):
            if attention_tokens.shape[1] != f:
                raise ValueError(
                    f"Attention tokens feature dim {attention_tokens.shape[1]} "
                    f"must match x_cmp feature dim {f}"
                )
            db_complex_tensors.append(
                attention_tokens.to(x_cmp.device)
            )
            db_reim_tensors_for_faiss.append(
                th.cat(
                    [
                        attention_tokens.real.to(x_cmp.device),
                        attention_tokens.imag.to(x_cmp.device),
                    ],
                    dim=1,
                )
            )

        full_db_complex_tensor = th.cat(db_complex_tensors, dim=0)
        db_for_faiss_reim = th.cat(db_reim_tensors_for_faiss, dim=0)

        if (
            db_for_faiss_reim.shape[0] == 0
        ):  # Should not happen if b_query > 0
            return x_cmp

        db_for_faiss_np = (
            db_for_faiss_reim.detach()
            .cpu()
            .contiguous()
            .float()
            .numpy()
        )
        faiss.normalize_L2(
            db_for_faiss_np
        )  # Normalize the entire database for IndexFlatIP

        index = faiss.IndexFlatIP(
            2 * f
        )  # 2*f because we concat real and imag
        index.add(db_for_faiss_np)

        # --- 2. Prepare query vectors (current batch features) ---
        query_reim = th.cat([x_cmp.real, x_cmp.imag], dim=1)
        query_np = (
            query_reim.detach().cpu().contiguous().float().numpy()
        )
        faiss.normalize_L2(query_np)  # Normalize queries separately

        # Determine actual k for search (cannot be more than num items in index)
        actual_top_k = min(self.top_k, db_for_faiss_np.shape[0])
        if actual_top_k == 0:  # No items to search or k=0
            return x_cmp

        _, nn_idx = index.search(
            query_np, actual_top_k
        )  # nn_idx shape: (b_query, actual_top_k)

        # --- 3. Neighbour aggregation in polar coords ---
        # Gather complex neighbours from the full_db_complex_tensor using nn_idx
        nbrs = full_db_complex_tensor[
            nn_idx
        ]  # Advanced indexing keeps complex dtype

        rho = th.abs(nbrs) + 1e-7  # Add eps for stability
        phi = th.angle(nbrs)

        mean_rho = rho.mean(dim=1)
        mean_phi = th.atan2(
            th.sin(phi).mean(dim=1),  # "circular mean"
            th.cos(phi).mean(dim=1),
        )
        agg = mean_rho * th.exp(1j * mean_phi)  # (B_query, F) complex

        # --- 4. Residual connection ---
        current_alpha = self.alpha
        if isinstance(self.alpha, nn.Parameter):
            current_alpha = th.clamp(self.alpha, 0.0, 1.0)

        out = (1.0 - current_alpha) * x_cmp + current_alpha * agg
        return out


# ------------------------------------------------------------------
# 3. Core RG‑AEG modules
# ------------------------------------------------------------------
class OptAEGV3(nn.Module):
    def __init__(
        self,
        use_sparse=False,
        top_k=32,
        init_alpha: float = 0.2,
        learnable_alpha: bool = True,
        num_attention_tokens: int = 0,
        flat_features_dim: int = 0,
    ):
        super().__init__()
        self.use_sparse = use_sparse
        self.sparse = (
            SparseRetriever(
                top_k=top_k,
                init_alpha=init_alpha,
                learnable_alpha=learnable_alpha,
            )
            if use_sparse
            else None
        )

        self.attention_tokens = None
        if use_sparse and num_attention_tokens > 0:
            if flat_features_dim <= 0:
                raise ValueError(
                    "flat_features_dim must be positive if num_attention_tokens > 0"
                )
            # Initialize attention tokens with small random complex values
            # A slightly larger initial magnitude might encourage them to act as attractors earlier.
            # Let's try random phase and a modest initial magnitude.
            init_mag = 0.1
            magnitudes = (
                th.ones(num_attention_tokens, flat_features_dim)
                * init_mag
            )
            phases = (
                th.rand(num_attention_tokens, flat_features_dim)
                * 2
                * math.pi
            )
            self.attention_tokens = nn.Parameter(
                th.polar(magnitudes, phases.to(th.complex64))
            )
            # self.attention_tokens = nn.Parameter(
            #    (th.randn(num_attention_tokens, flat_features_dim, dtype=th.complex64)) * 0.05
            # )
            print(
                f"Initialized {num_attention_tokens} attention tokens with dim {flat_features_dim}"
            )

        c64 = th.complex64
        self.vx = nn.Parameter(th.rand(1, 1, 1, dtype=c64) / 100)
        self.vy = nn.Parameter(th.rand(1, 1, 1, dtype=c64) / 100)
        self.wx = nn.Parameter(th.rand(1, 1, 1, dtype=c64) / 100)
        self.wy = nn.Parameter(th.rand(1, 1, 1, dtype=c64) / 100)
        self.afactor = nn.Parameter(th.rand(1, 1, dtype=c64) / 100)
        self.mfactor = nn.Parameter(th.rand(1, 1, dtype=c64) / 100)

    @staticmethod
    def flow(dx, dy, x):
        return x * (1 + dy) + dx

    @staticmethod
    def normalize(z):
        eps = 1e-8
        rho = th.abs(z) + eps
        theta = th.atan2(
            th.imag(z) + eps, th.real(z) + eps
        )  # Added eps to real for atan2 stability at origin
        unit = th.cos(theta) + 1j * th.sin(theta)
        return th.tanh(rho) * unit

    def forward(self, data):  # data is x_cmp from RG_AEG_Block
        shp = data.size()  # (B, C_complex, H_feat, W_feat)
        data_flat = data.flatten(
            1
        )  # (B, C_complex * H_feat * W_feat) -> (B, F_flat)
        data_norm = self.normalize(data_flat)

        if self.use_sparse and self.sparse is not None:
            # Pass attention_tokens if they exist and are on the correct device
            attn_tokens_to_pass = self.attention_tokens
            if attn_tokens_to_pass is not None:
                attn_tokens_to_pass = attn_tokens_to_pass.to(
                    data_norm.device
                )
            data_norm = self.sparse(
                data_norm, attention_tokens=attn_tokens_to_pass
            )

        b = shp[0]  # Original batch size
        # view expects (B, F_flat), so data_norm is already (B, F_flat)
        # The -1 in view(b, -1, 1) implies F_flat. So view should be (b, F_flat, 1)
        view = data_norm.view(b, data_norm.shape[1], 1)

        v = self.flow(self.vx, self.vy, view)
        w = self.flow(self.wx, self.wy, view)

        dx = self.afactor * (v * th.tanh(w)).squeeze(
            -1
        )  # dx should be (B, F_flat)
        dy = self.mfactor * th.tanh(
            data_norm
        )  # dy should be (B, F_flat)

        data_flow = self.flow(dx, dy, data_norm)
        data_final = self.normalize(data_flow)
        return data_final.view(
            *shp
        )  # Reshape back to (B, C_complex, H_feat, W_feat)


class RG_AEG_Block(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=2,
        use_sparse=False,
        top_k=32,
        init_alpha: float = 0.2,
        learnable_alpha: bool = True,
        input_h: int = 0,
        input_w: int = 0,
        num_attention_tokens: int = 0,
    ):  # Added input_h, input_w
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch * 2, kernel_size=4, stride=stride, padding=1
        )
        self.norm = NormalityNormalization(
            out_ch * 2,
            eps=1e-4,
            elementwise_affine=True,
            noise_factor=0.5,
        )
        self.out_ch = out_ch  # Number of complex channels
        self.stride = stride

        flat_features_dim_for_aeg = 0
        if use_sparse and num_attention_tokens > 0:
            if input_h <= 0 or input_w <= 0:
                raise ValueError(
                    "input_h and input_w must be positive for attention token dimension calculation."
                )
            # Calculate spatial dimensions after convolution
            # K=4, P=1, S=stride
            h_after_conv = (
                math.floor((input_h - 4 + 2 * 1) / stride) + 1
            )
            w_after_conv = (
                math.floor((input_w - 4 + 2 * 1) / stride) + 1
            )
            flat_features_dim_for_aeg = (
                self.out_ch * h_after_conv * w_after_conv
            )
            if flat_features_dim_for_aeg <= 0:
                raise ValueError(
                    f"Calculated flat_features_dim_for_aeg is {flat_features_dim_for_aeg}, must be positive."
                )

        self.aeg = OptAEGV3(
            use_sparse=use_sparse,
            top_k=top_k,
            init_alpha=init_alpha,
            learnable_alpha=learnable_alpha,
            num_attention_tokens=num_attention_tokens,
            flat_features_dim=flat_features_dim_for_aeg,
        )

    def forward(self, x):
        if x.is_complex():  # Ensure input is real for Conv2d
            x = th.cat([x.real, x.imag], dim=1)

        x_conv = self.conv(x)  # (B, out_ch*2, H_conv, W_conv)
        x_norm = self.norm(x_conv)

        # Split into real and imaginary parts for complex tensor
        # x_norm is (B, out_ch*2, H_conv, W_conv)
        # x_cmp should be (B, out_ch, H_conv, W_conv)
        x_cmp = (
            x_norm[:, : self.out_ch] + 1j * x_norm[:, self.out_ch :]
        )

        x_aeg = self.aeg(
            x_cmp
        )  # complex output, (B, out_ch, H_conv, W_conv)
        x_out = th.cat(
            [x_aeg.real, x_aeg.imag], dim=1
        )  # (B, out_ch*2, H_conv, W_conv)
        return x_aeg, x_out


class RG_AEG_Net(nn.Module):
    def __init__(self, num_attention_tokens: int = 0):
        super().__init__()
        self.num_attention_tokens = num_attention_tokens

        # Block 1: No sparse retrieval, so attention tokens are not relevant here.
        self.block1 = RG_AEG_Block(
            in_ch=1,
            out_ch=4,
            stride=2,
            use_sparse=False,
            input_h=28,
            input_w=28,
            num_attention_tokens=0,
        )
        # Even if num_attention_tokens > 0, it won't be used as use_sparse=False

        # Calculate output spatial dimensions from block1 for block2's input_h, input_w
        h_b1_out = math.floor((28 - 4 + 2 * 1) / 2) + 1  # 14
        w_b1_out = math.floor((28 - 4 + 2 * 1) / 2) + 1  # 14

        # Block 2: Uses sparse retrieval. Pass num_attention_tokens.
        # Input channels for block2's conv is block1.out_ch * 2 (real channels)
        self.block2 = RG_AEG_Block(
            in_ch=4 * 2,
            out_ch=8,
            stride=2,
            use_sparse=True,
            top_k=48,
            init_alpha=0.015,
            learnable_alpha=True,
            input_h=h_b1_out,
            input_w=w_b1_out,
            num_attention_tokens=self.num_attention_tokens,
        )

        # Calculate output spatial dimensions from block2 for block3's input_h, input_w
        h_b2_out = math.floor((h_b1_out - 4 + 2 * 1) / 2) + 1  # 7
        w_b2_out = math.floor((w_b1_out - 4 + 2 * 1) / 2) + 1  # 7

        # Block 3: Uses sparse retrieval. Pass num_attention_tokens.
        # Input channels for block3's conv is block2.out_ch * 2 (real channels)
        self.block3 = RG_AEG_Block(
            in_ch=8 * 2,
            out_ch=8,
            stride=2,
            use_sparse=True,
            top_k=48,
            init_alpha=0.015,
            learnable_alpha=True,
            input_h=h_b2_out,
            input_w=w_b2_out,
            num_attention_tokens=self.num_attention_tokens,
        )

        # FC layer input: block3 outputs x_out which is (B, block3.out_ch*2, H_final, W_final)
        # H_final for FC layer is output of block3.conv
        h_b3_out = math.floor((h_b2_out - 4 + 2 * 1) / 2) + 1  # 3
        w_b3_out = math.floor((w_b2_out - 4 + 2 * 1) / 2) + 1  # 3
        fc_input_features = (
            (8 * 2) * h_b3_out * w_b3_out
        )  # 16 * 3 * 3 = 144
        self.fc = nn.Linear(fc_input_features, 10)

    def forward(self, x):
        _, r1 = self.block1(
            x
        )  # r1 is real tensor (B, block1.out_ch*2, H, W)
        _, r2 = self.block2(r1)  # r2 is real tensor
        _, r3 = self.block3(r2)  # r3 is real tensor
        flat = th.flatten(r3, 1)
        return self.fc(flat)


# ------------------------------------------------------------------
# 4. Training / evaluation helpers
# ------------------------------------------------------------------
def count_params(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def train_epoch(model, loader, opt, crit, device):
    model.train()
    t_loss, correct, total = 0.0, 0, 0
    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        out = model(data)
        loss = crit(out, target)

        if not th.isfinite(loss):
            print(
                f"Warning: Non-finite loss detected: {loss.item()}. Skipping batch."
            )
            th.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.5
            )  # Still clip
            continue  # Skip opt.step()

        loss.backward()
        total_norm = th.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=0.5
        )

        if not th.isfinite(total_norm):
            print(
                f"Warning: Non-finite gradient norm: {total_norm}. Grads may be unstable."
            )
            # Optionally, zero out gradients if norm is bad before step
            # opt.zero_grad()
            # continue

        opt.step()
        t_loss += loss.item() * data.size(0)
        pred = out.argmax(1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    if total == 0:
        return 0.0, 0.0
    return t_loss / total, 100 * correct / total


@th.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    t_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss = crit(out, target)
        t_loss += loss.item() * data.size(0)
        pred = out.argmax(1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    if total == 0:
        return 0.0, 0.0
    return t_loss / total, 100 * correct / total


# ------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------
def main(cfg):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_ld = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )  # drop_last for stable batch sizes
    test_ld = DataLoader(
        test_ds,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = RG_AEG_Net(
        num_attention_tokens=cfg.num_attention_tokens
    ).to(device)
    print(model)
    print(f"Params: {count_params(model):,}")
    if cfg.num_attention_tokens > 0:
        print(
            f"Using {cfg.num_attention_tokens} global attention tokens per sparse block."
        )

    optim = th.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=1e-2
    )
    sched = th.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.epochs
    )
    crit = nn.CrossEntropyLoss(label_smoothing=0.0)

    best_acc = 0.0
    for ep in range(1, cfg.epochs + 1):
        st = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_ld, optim, crit, device
        )
        te_loss, te_acc = evaluate(model, test_ld, crit, device)
        sched.step()
        if te_acc > best_acc:
            best_acc = te_acc
            th.save(model.state_dict(), cfg.out / "best.pt")

        log_msg_parts = [
            f"[{ep:02}/{cfg.epochs}]",
            f"train {tr_loss:.3f}/{tr_acc:.2f}%",
            f"test {te_loss:.3f}/{te_acc:.2f}%",
            f"lr {optim.param_groups[0]['lr']:.1e}",
        ]

        # Log alpha and attention token norms
        if cfg.sparse:  # This flag enables sparse in general
            alpha_str_parts = []
            attn_token_norm_parts = []

            for block_name, block in [
                ("block2", model.block2),
                ("block3", model.block3),
            ]:
                if (
                    hasattr(block.aeg, "sparse")
                    and block.aeg.sparse is not None
                ):
                    alpha_val = block.aeg.sparse.alpha
                    if isinstance(alpha_val, nn.Parameter):
                        alpha_val = alpha_val.item()
                    alpha_str_parts.append(
                        f"{block_name}_alpha={alpha_val:.3f}"
                    )

                if (
                    hasattr(block.aeg, "attention_tokens")
                    and block.aeg.attention_tokens is not None
                ):
                    # Avg L2 norm of attention tokens (magnitude part)
                    with th.no_grad():
                        token_mags = th.abs(
                            block.aeg.attention_tokens
                        )
                        avg_norm = (
                            token_mags.norm(p=2, dim=1).mean().item()
                        )
                        # Or just mean magnitude: avg_norm = token_mags.mean().item()
                    attn_token_norm_parts.append(
                        f"{block_name}_atn_norm={avg_norm:.3f}"
                    )

            if alpha_str_parts:
                log_msg_parts.append(
                    f"Alphas: {', '.join(alpha_str_parts)}"
                )
            if attn_token_norm_parts:
                log_msg_parts.append(
                    f"AttnTokenNorms: {', '.join(attn_token_norm_parts)}"
                )

        log_msg_parts.append(f"{time.time() - st:.1f}s")
        print(" | ".join(log_msg_parts))

    print(f"Best test accuracy: {best_acc:.2f}%")
    th.save(model.state_dict(), cfg.out / "last.pt")


# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RG‑AEG × RetroInfer demo with Opt. Attention Tokens"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="training epochs"
    )
    parser.add_argument(
        "--bs", type=int, default=128, help="batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate"
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="enable RetroInfer sparse retrieval generally",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=32,
        help="top‑k neighbours to retrieve",
    )
    parser.add_argument(
        "--num_attention_tokens",
        type=int,
        default=0,
        help="Number of global attention tokens for sparse blocks (0 to disable)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("./checkpoints")
    )
    args = parser.parse_args()
    print(args)

    # Ensure sparse is implicitly true if attention tokens are requested
    if args.num_attention_tokens > 0 and not args.sparse:
        print(
            "Enabling --sparse mode because --num_attention_tokens > 0."
        )
        args.sparse = True

    args.out.mkdir(exist_ok=True, parents=True)
    main(args)
