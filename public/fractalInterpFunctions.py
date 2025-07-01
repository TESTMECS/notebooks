# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets==3.6.0",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.0",
#     "polars==1.30.0",
#     "torch==2.7.1",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(
    width="medium",
    layout_file="layouts/fractalInterpFunctions.slides.json",
)


@app.cell
def _():
    import re
    from collections import Counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from datasets import load_dataset
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    return Adam, Counter, DataLoader, load_dataset, mo, nn, np, plt, re, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🌈 Fractal Interpolation Functions & Neural Applications 🤖

    This notebook explores the fascinating world of fractal interpolation using the Read-Bajraktarević (R-B) operator. We'll journey from simple linear base functions to more complex neural network bases, culminating in an exciting application: using fractal embeddings as positional encodings in a Transformer model! 🚀

    ## 🌟 Overview

    Fractal interpolation generates complex, self-similar curves that pass through a given set of data points. Unlike traditional interpolation methods that produce smooth functions, fractal interpolation can capture intricate, "rough" details, making it ideal for modeling natural phenomena. The core idea relies on an Iterated Function System (IFS), where the fractal interpolant is the fixed point (attractor) of a special operator.

    This notebook demonstrates:

    - Linear Base Interpolation: A fundamental example using a simple linear function as the base interpolant. 📏

    - Neural Base Interpolation: Replacing the linear base with a small neural network to learn a more flexible base interpolant. 🧠

    - α-Fractal Positional Encoding: Applying the fractal interpolation concept to create positional embeddings for Transformer models, comparing its performance against traditional sinusoidal positional encoding on a text classification task. 📊

    ## 💡 Key Concepts

    - Read-Bajraktarević (R-B) Operator: A contractive operator whose fixed point is the fractal interpolant. It defines how the function at a point depends on its value at a "scaled-down" version of that point. 🔄

    - Base Function (β): A standard interpolant (e.g., linear, polynomial) that passes through the data points. The fractal interpolant is built upon this base.

    - Scaling Factor (α): A crucial parameter (between 0 and 1) that controls the "fractalness" or "roughness" of the interpolant. A higher α generally leads to more fractal-like behavior. 🏔️

    - Iterative Approximation: The fixed point of the R-B operator is found by repeatedly applying the operator to an initial guess (often the base function itself). 🔄
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 📈 1. Linear Base Interpolation

    In this section, we start with a classic example: interpolating f(x) = sin(2πx) using a linear base function.

    We define β(x) as a simple piecewise linear interpolant.

    The inv_L function computes the inverse affine transformation for each subinterval, mapping a point x back to the original [a,b] domain.

    The T operator is constructed, combining β, inv_L, and the scaling factor α.

    We then iteratively apply T to β until φ (our fractal interpolant) converges.

    The resulting plot clearly shows how the fractal interpolant passes through the original nodes while exhibiting more roughness compared to a pure linear interpolation. 📉✨
    """
    )
    return


@app.cell
def _(mo, np, plt):
    def _():
        # 1. Problem setup
        f = lambda x: np.sin(2 * np.pi * x)
        a, b = 0.0, 1.0
        N = 5  # number of intervals
        nodes = np.linspace(a, b, N + 1)

        # 2. Linear base interpolant β
        def beta(x):
            # find interval
            i = np.clip(np.searchsorted(nodes, x) - 1, 0, N - 1)
            x0, x1 = nodes[i], nodes[i + 1]
            y0, y1 = f(x0), f(x1)
            return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

        # 3. Define Li⁻¹ on each subinterval
        def inv_L(i, x):
            # since Li(x) = a_i x + b_i is simple affine mapping [a,b]→[x_{i-1},x_i]
            return ((x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])) * (b - a) + a

        # 4. Build the operator T
        alpha = 0.5

        def T(phi, xs):
            ys = np.zeros_like(xs)
            for idx, x in enumerate(xs):
                # locate i
                i = np.clip(np.searchsorted(nodes, x) - 1, 0, N - 1)
                x_prev = inv_L(i + 1, x)
                ys[idx] = alpha * phi(x_prev) + f(x) - alpha * beta(x_prev)
            return ys

        # 5. Iterate to approximate fixed point
        xs = np.linspace(a, b, 500)
        phi_vals = beta(xs)  # initial guess = β
        for _ in range(10):
            phi_vals = T(lambda x: np.interp(x, xs, phi_vals), xs)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, f(xs), label="f(x) = sin(2πx)")
        ax.plot(xs, phi_vals, label=f"α-fractal approx (α={alpha})")
        ax.scatter(nodes, f(nodes), color="k", zorder=5)
        ax.legend()
        ax.set_title("α-Fractal Interpolation with Linear Base")
        mo.output.append(fig)


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 🧠 2. Neural Base Interpolation

    Taking things up a notch, we replace the fixed linear base function with a small neural network (an MLP) that learns to interpolate the original data points.

    A BaseNet (a simple nn.Sequential model) is defined and trained to minimize the MSE loss on the given nodes (nodes, f(nodes)). 🚀

    Once trained, this BaseNet becomes our β for the R-B operator.

    The T operator is then applied iteratively, similar to the linear case, but now leveraging the learned neural base.

    This demonstrates the flexibility of using neural networks to define the base, potentially allowing for more complex or adaptive fractal interpolants. 🎨🤖
    """
    )
    return


@app.cell
def _(Adam, mo, nn, np, plt, torch):
    def _():
        # 1. Problem setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f = lambda x: np.sin(2 * np.pi * x)
        a, b = 0.0, 1.0
        N = 8
        nodes = np.linspace(a, b, N + 1)

        # 2. Define & train a small MLP β to interpolate f at the nodes
        class BaseNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.net(x)

        # prepare training data (nodes)
        x_train = torch.tensor(nodes, dtype=torch.float32, device=device).unsqueeze(1)
        y_train = torch.tensor(f(nodes), dtype=torch.float32, device=device).unsqueeze(
            1
        )

        beta = BaseNet().to(device)
        opt = Adam(beta.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        # train for a quick fit
        for epoch in mo.status.progress_bar(range(500)):
            opt.zero_grad()
            pred = beta(x_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            opt.step()
        # β is now our learned interpolant

        # 3. Read–Bajraktarević operator T using α
        alpha = 0.6
        xs = np.linspace(a, b, 500)
        xs_t = torch.tensor(xs, dtype=torch.float32, device=device).unsqueeze(1)

        # Precompute: interval widths & inverses
        widths = nodes[1:] - nodes[:-1]

        def inv_L(i, x):
            # map x back to [a,b]
            return (x - nodes[i]) / widths[i] * (b - a) + a

        with torch.no_grad():
            # initial φ = β(xs)
            phi = beta(xs_t).squeeze().cpu().numpy()

        for _ in range(15):
            phi_prev = phi.copy()
            phi_tensor = torch.tensor(phi_prev, dtype=torch.float32, device=device)
            new_vals = []
            for x in xs:
                # find interval i
                i = min(np.searchsorted(nodes, x) - 1, N - 1)
                x0 = inv_L(i, x)
                # network inputs must be normalized to [a,b]
                inp = torch.tensor([[x0]], device=device, dtype=torch.float32)
                b_val = beta(inp).item()
                phi_val = float(
                    alpha * np.interp(x0, xs, phi_prev) + f(x) - alpha * b_val
                )
                new_vals.append(phi_val)
            phi = np.array(new_vals)

        fig, ax = plt.subplots(figsize=(8, 4)) # Create a figure and a set of subplots

        ax.plot(xs, f(xs), label="Original f(x)=sin(2πx)")
        ax.plot(xs, phi, label=f"α-Fractal (α={alpha})")
        ax.scatter(nodes, f(nodes), color="k", zorder=5, label="Nodes")

        ax.legend()
        ax.set_title("Neural-Based α-Fractal Interpolation")
        mo.output.append(fig)
        return
    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 📊 3. α-Fractal vs. Sinusoidal Positional Encoding

    This is where the fractal interpolation concept meets the cutting edge of deep learning! Positional encodings are vital in Transformers to infuse sequence order information. Traditionally, sinusoidal functions are used. Here, we explore using a fractal-based approach.

    We load the ag_news text classification dataset. 📰

    A basic tokenizer and vocabulary are created.

    The FractalPE module is implemented, creating positional embeddings based on the R-B operator, with trainable "anchor" points.

    A simple EncoderClassifier (a small Transformer encoder) is built, which can switch between traditional sinusoidal positional encoding and our new FractalPE.

    Both models are trained for a single epoch (for a quick demo) and evaluated.

    The output shows a comparison of training and test accuracies. You'll also see a visualization of some dimensions of the generated FractalPE vectors, highlighting their unique patterns! 📈🔥
    """
    )
    return


@app.cell
def _(mo):
    cfg = {
        "batch_size": mo.ui.slider(4, 64, value=16, step=4, label="Batch Size"),
        "d_model": mo.ui.slider(32, 256, value=64, step=32, label="Model Dimension"),
        "nhead": mo.ui.slider(2, 8, value=4, step=2, label="Attention Heads"),
        "layers": mo.ui.slider(1, 4, value=2, step=1, label="Transformer Layers"),
        "train_samples": mo.ui.slider(500, 5000, value=1000, step=500, label="Training Samples"),
        "test_samples": mo.ui.slider(200, 2000, value=400, step=200, label="Test Samples"),
        "max_len": mo.ui.slider(64, 256, value=128, step=32, label="Max Sequence Length"),
        "fractal_segments": mo.ui.slider(4, 16, value=8, step=2, label="Fractal Segments"),
        "fractal_alpha": mo.ui.slider(0.1, 0.9, value=0.6, step=0.1, label="Fractal Alpha"),
        "fractal_iters": mo.ui.slider(2, 8, value=4, step=1, label="Fractal Iterations"),
        "learning_rate": mo.ui.slider(1e-4, 1e-2, value=3e-4, step=1e-4, label="Learning Rate"),
        "clear_cache_freq": mo.ui.checkbox(value=True, label="Clear VRAM Cache Frequently")
    }

    # Display current configuration
    config_display = mo.vstack([
        mo.md("### Configuration Panel (Running this cell resets to defaults)"),
        mo.hstack([
            mo.vstack([
                cfg["batch_size"],
                cfg["d_model"], 
                cfg["nhead"],
                cfg["layers"]
            ]),
            mo.vstack([
                cfg["train_samples"],
                cfg["test_samples"],
                cfg["max_len"],
                cfg["learning_rate"]
            ]),
            mo.vstack([
                cfg["fractal_segments"],
                cfg["fractal_alpha"],
                cfg["fractal_iters"],
                cfg["clear_cache_freq"]
            ])
        ])
    ])

    mo.output.append(config_display)
    return (cfg,)


@app.cell
def _(Counter, DataLoader, cfg, load_dataset, mo, nn, plt, re, torch):
    def _():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(69420)
        mo.output.append(mo.md("### Running with Configuration:"))
        with mo.redirect_stdout():
            print([f"{k}: {v.value}" for k, v in cfg.items()])

        # Extract config values
        BATCH_SIZE = cfg["batch_size"].value
        D_MODEL = cfg["d_model"].value
        NHEAD = cfg["nhead"].value
        LAYERS = cfg["layers"].value
        TRAIN_SAMPLES = cfg["train_samples"].value
        TEST_SAMPLES = cfg["test_samples"].value
        MAX_LEN = cfg["max_len"].value
        FRACTAL_SEGMENTS = cfg["fractal_segments"].value
        FRACTAL_ALPHA = cfg["fractal_alpha"].value
        FRACTAL_ITERS = cfg["fractal_iters"].value
        LEARNING_RATE = cfg["learning_rate"].value
        CLEAR_CACHE_FREQ = cfg["clear_cache_freq"].value

        # Clear any existing VRAM usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ────────────────── 1. Data ──────────────────
        # Training on AG News
        ds = load_dataset("ag_news")  # train/test splits, 4 labels (0‑3)
        # simple "basic_english" tokenizer (whitespace + punctuation)
        TOKEN_RE = re.compile(r"\w+|[^\w\s]")
        def tokenize(txt: str):
            return TOKEN_RE.findall(txt.lower())
        # build a tiny vocab (min_freq = 2 keeps the demo light)
        counter = Counter(tok for ex in ds["train"]["text"] for tok in tokenize(ex))
        PAD, UNK = "<pad>", "<unk>"
        itos = [PAD, UNK] + [t for t, c in counter.items() if c >= 2]
        stoi = {t: i for i, t in enumerate(itos)}
        def numericalise(tokens):
            return [stoi.get(t, stoi[UNK]) for t in tokens]
        PAD_IDX, NUM_CLASSES = stoi[PAD], 4
        def collate(batch):
            toks = [numericalise(tokenize(ex["text"]))[:MAX_LEN] for ex in batch]
            lbls = torch.tensor([ex["label"] for ex in batch])
            lens = [len(t) for t in toks]
            padded = torch.full((len(toks), MAX_LEN), PAD_IDX, dtype=torch.long)
            for i, t in enumerate(toks):
                padded[i, : lens[i]] = torch.tensor(t, dtype=torch.long)
            attn = padded != PAD_IDX
            return padded.to(device), attn.to(device), lbls.to(device)
        # Use configurable data loader settings
        train_loader = DataLoader(
            ds["train"].select(range(min(TRAIN_SAMPLES, len(ds["train"])))),
            shuffle=True,
            batch_size=BATCH_SIZE,
            collate_fn=collate,
            num_workers=0,
            pin_memory=False,
        )
        test_loader = DataLoader(
            ds["test"].select(range(min(TEST_SAMPLES, len(ds["test"])))),
            batch_size=BATCH_SIZE, 
            collate_fn=collate, 
            num_workers=0, 
            pin_memory=False
        )
        # ────────────────── 2. Fractal PE ──────────────────
        class FractalPE(nn.Module):
            """Uniform‑grid α‑fractal positional encoding (1‑D Read–Bajraktarević)."""
            def __init__(self, d_model, max_len, segments=FRACTAL_SEGMENTS, alpha=FRACTAL_ALPHA, iters=FRACTAL_ITERS):
                super().__init__()
                self.alpha, self.iters, self.seg = alpha, iters, segments
                self.anchors = nn.Parameter(torch.randn(segments + 1, d_model))

                t = torch.linspace(0, 1, max_len)
                seg_id = torch.floor(t * segments).long()
                lin_w = t * segments - seg_id.float()
                self.register_buffer("seg_id", seg_id)
                self.register_buffer("lin_w", lin_w.unsqueeze(-1))
            def forward(self, L):
                L = min(L, self.seg_id.shape[0])
                anchors_idx0 = self.seg_id[:L]
                anchors_idx1 = (self.seg_id[:L] + 1).clamp(max=self.seg)
                base = (1 - self.lin_w[:L]) * self.anchors[anchors_idx0] + self.lin_w[
                    :L
                ] * self.anchors[anchors_idx1]
                phi = base
                for _ in range(self.iters):
                    x_local = (torch.linspace(0, 1, L, device=phi.device) * self.seg) % 1
                    idx = torch.round(x_local * (L - 1)).long()
                    phi = self.alpha * phi[idx] + (1 - self.alpha) * base
                return phi
        # ────────────────── 3. Encoder model ──────────────────
        class EncoderClassifier(nn.Module):
            def __init__(self, d_model=D_MODEL, nhead=NHEAD, layers=LAYERS, use_fractal=False):
                super().__init__()
                self.emb = nn.Embedding(len(itos), d_model, padding_idx=PAD_IDX)
                self.pos = (
                    FractalPE(d_model, MAX_LEN)
                    if use_fractal
                    else nn.Embedding(MAX_LEN, d_model)
                )
                # Configurable FFN dimension for memory efficiency
                enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc, layers)
                self.fc = nn.Linear(d_model, NUM_CLASSES)
                self.use_fractal = use_fractal
            def forward(self, x, mask):
                h = self.emb(x)

                if self.use_fractal:
                    pos_enc = self.pos(x.size(1))
                else:
                    pos_enc = self.pos(torch.arange(x.size(1), device=x.device))

                h = h + pos_enc
                h = self.encoder(h, src_key_padding_mask=~mask)
                out = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                return self.fc(out)
        # ────────────────── 4. Train / evaluate helpers ──────────────────
        def epoch_loop(model, loader, opt=None):
            train = opt is not None
            model.train() if train else model.eval()
            tot = correct = 0

            with torch.set_grad_enabled(train):
                for x, m, y in loader:

                    if train:
                        opt.zero_grad()

                    logits = model(x, m)

                    if train:
                        loss = nn.CrossEntropyLoss()(logits, y)
                        loss.backward()
                        opt.step()
                        # Clear intermediate tensors
                        del loss

                    pred = logits.argmax(1)
                    tot += y.size(0)
                    correct += (pred == y).sum().item()

                    # Clear batch tensors from VRAM (configurable)
                    del x, m, y, logits, pred
                    if CLEAR_CACHE_FREQ and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return correct / tot
        def run(use_fractal):
            # Clear VRAM before creating new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = EncoderClassifier(use_fractal=use_fractal).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            train_acc = epoch_loop(model, train_loader, opt)
            test_acc = epoch_loop(model, test_loader)

            # Clean up model from memory
            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return train_acc, test_acc

        sin_train, sin_test = run(False)
        fra_train, fra_test = run(True)
        mo.output.append(mo.md("### Results:"))
        mo.output.append(f"Sinusoidal PE – train {sin_train * 100:.1f}% | test {sin_test * 100:.1f}%")
        mo.output.append(f"Fractal    PE – train {fra_train * 100:.1f}% | test {fra_test * 100:.1f}%")
        mo.output.append(f"Sinusoidal PE – train {sin_train*100:.1f}% | test {sin_test*100:.1f}%")
        mo.output.append(f"Fractal    PE – train {fra_train*100:.1f}% | test {fra_test*100:.1f}%")
        fractal_pe_module = FractalPE(
            d_model=D_MODEL, max_len=MAX_LEN, segments=FRACTAL_SEGMENTS, alpha=FRACTAL_ALPHA, iters=FRACTAL_ITERS
        ).to(device)
        pe_vectors = fractal_pe_module(MAX_LEN).detach().cpu().numpy()
        #
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(4):  # Plot first 4 dimensions
            ax.plot(pe_vectors[:, i], label=f"Dim {i}")
        ax.set_title("Sample Fractal PE Dimensions")
        ax.set_xlabel("Position")
        ax.set_ylabel("Value")
        ax.legend()
        mo.output.append(fig)
    _()
    return


if __name__ == "__main__":
    app.run()
