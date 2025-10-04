# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets==3.6.0",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
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
    # 🧠 Core Idea

    The goal is to check whether amino acid sequences and their 3D structure strings (3Di) encode consistent, symmetric information — like a "mirror test" for biological data.

    ## 🧬 Dataset

        ProstT5Dataset: Contains tokenized amino acid sequences (input_id_y) and their corresponding 3Di structure tokens (input_id_x), extracted from the AlphaFold DB.

        Each sample is used to generate a positive pair (real match) and a negative pair (random mismatch) to train a classifier.

    ## 🏗️ Model Architecture: ChiralTranslatorNet

        Embeds each input (sequence & structure) → then processes both:

            Left Path: Sequence ➝ Structure

            Right Path: Structure ➝ Sequence

        These two paths are run through the same CNN.

        Final output is the difference between left and right pathways → passed to a classifier.

    > 💡 Think of it like a symmetry detector — is the model’s representation similar in both directions?

    # 🧪 Hypothesis:
        Since AlphaFold’s 3Di structures are learned approximations of the true spatial configuration of a protein, and are derived deterministically from the amino acid sequence, there exists a translation symmetry between the sequence and its structure — i.e., the learned representations should be bi-directionally consistent and the ChiralNet should pick up on this pattern.
    """
    )
    return


@app.cell
def _():
    # --- Protein Translation Symmetry Test ---
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_dataset
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    import marimo as mo

    return DataLoader, Dataset, load_dataset, mo, nn, plt, random, torch


@app.cell
def _(torch):
    # --- Configuration ---
    MAX_LENGTH = 256  # Pad all sequences to a fixed length
    VOCAB_SIZE = 31  # ProstT5 vocabulary size for both AA and 3Di
    EPOCHS = 10
    BATCH_SIZE = 64  # Smaller batch size due to larger model/data
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Protein Translation Symmetry Test ---")
    print(f"Using device: {DEVICE}")
    return BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, MAX_LENGTH, VOCAB_SIZE


@app.cell
def _(Dataset, VOCAB_SIZE, load_dataset, mo, random, torch):
    # --- Custom Dataset for Protein Translation Pairs ---
    class ProteinTranslationDataset(Dataset):
        def __init__(self, split="train", max_length=256, subset_size=None):
            self.max_length = max_length
            mo.output.append(f"Loading and processing '{split}' data...")
            dataset = load_dataset("Rostlab/ProstT5Dataset", split=split)

            if subset_size:
                dataset = dataset.select(range(subset_size))

            self.sequences = dataset["input_id_y"]
            self.structures = dataset["input_id_x"]
            self.num_samples = len(self.sequences)
            self.vocab_size = VOCAB_SIZE

        def __len__(self):
            return (
                self.num_samples * 2
            )  # We create one positive and one negative sample for each entry

        def __getitem__(self, idx):
            # Determine if this should be a positive (match) or negative (mismatch) sample
            is_match = idx % 2 == 0
            original_idx = idx // 2

            seq = self.sequences[original_idx]

            if is_match:
                struct = self.structures[original_idx]
                label = 1.0
            else:
                # Create a mismatch
                mismatch_idx = random.randint(0, self.num_samples - 1)
                # Ensure it's a true mismatch
                while mismatch_idx == original_idx:
                    mismatch_idx = random.randint(0, self.num_samples - 1)
                struct = self.structures[mismatch_idx]
                label = 0.0

            # Pad and TRUNCATE sequences to max_length
            # Take the first MAX_LENGTH elements if longer, otherwise pad
            seq_truncated = seq[: self.max_length]
            struct_truncated = struct[: self.max_length]

            seq_capped = [
                min(token_id, self.vocab_size - 1) for token_id in seq_truncated
            ]
            struct_capped = [
                min(token_id, self.vocab_size - 1) for token_id in struct_truncated
            ]
            seq_capped = [max(token_id, 0) for token_id in seq_capped]
            struct_capped = [max(token_id, 0) for token_id in struct_capped]

            seq_padded = torch.tensor(
                seq_capped + [0] * max(0, self.max_length - len(seq_truncated)),
                dtype=torch.long,
            )
            struct_padded = torch.tensor(
                struct_capped + [0] * max(0, self.max_length - len(struct_truncated)),
                dtype=torch.long,
            )

            return seq_padded, struct_padded, torch.tensor(label, dtype=torch.float32)

    return (ProteinTranslationDataset,)


@app.cell
def _(nn, torch):
    # --- Model Definition (Chiral CNN for Paired Sequences) ---
    class ChiralTranslatorNet(nn.Module):
        def __init__(self, vocab_size, embedding_dim=64, max_length=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

            # The input to the CNN will be the concatenated embeddings
            cnn_input_dim = embedding_dim * 2

            # A single, powerful CNN pathway shared by both directions
            # Adjusted kernel sizes and channels slightly for better feature extraction
            self.cnn_pathway = nn.Sequential(
                nn.Conv1d(
                    cnn_input_dim, 128, kernel_size=7, padding=3
                ),  # Increased kernel size
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 256, kernel_size=9, padding=4),  # Increased kernel size
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.AdaptiveAvgPool1d(1),  # Pool to get a fixed-size vector
            )
            self.output_layer = nn.Linear(
                256, 1
            )  # Output is a single logit for binary classification

        def forward(self, seq, struct):
            # Embed the sequences: [batch, max_length] -> [batch, max_length, embedding_dim]
            seq_embed = self.embedding(seq)
            struct_embed = self.embedding(struct)

            # --- Left Path: Sequence -> Structure ---
            # Concatenate embeddings: [batch, max_length, embedding_dim * 2]
            left_input = torch.cat([seq_embed, struct_embed], dim=2)
            # Transpose for Conv1d: [batch, channels, length]
            left_input_transposed = left_input.transpose(1, 2)
            l_out = self.cnn_pathway(left_input_transposed).squeeze(-1)

            # --- Right Path: Structure -> Sequence ---
            # Concatenate in reverse order
            right_input = torch.cat([struct_embed, seq_embed], dim=2)
            right_input_transposed = right_input.transpose(1, 2)
            r_out = self.cnn_pathway(right_input_transposed).squeeze(-1)

            # The competitive interaction
            net_difference = l_out - r_out
            final_output = self.output_layer(net_difference)

            return final_output, l_out.norm(), r_out.norm()
    return (ChiralTranslatorNet,)


@app.cell
def _(BATCH_SIZE, DataLoader, MAX_LENGTH, ProteinTranslationDataset):
    # --- Training and Evaluation ---
    train_dataset = ProteinTranslationDataset(
        split="train", max_length=MAX_LENGTH, subset_size=20000
    )
    validation_dataset = ProteinTranslationDataset(
        split="valid", max_length=MAX_LENGTH, subset_size=474
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, num_workers=2
    )
    return train_loader, validation_loader


@app.cell
def _(
    ChiralTranslatorNet,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    VOCAB_SIZE,
    mo,
    nn,
    torch,
    train_loader,
):
    model = ChiralTranslatorNet(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    mo.output.append("\n--- Starting Training on Translation Symmetry Task ---")
    left_norms, right_norms, losses = [], [], []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seq, struct, labels in train_loader:
            seq, struct, labels = (
                seq.to(DEVICE),
                struct.to(DEVICE),
                labels.to(DEVICE).unsqueeze(1),
            )

            optimizer.zero_grad()
            output, l_norm, r_norm = model(seq, struct)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        left_norms.append(l_norm.item())
        right_norms.append(r_norm.item())
        mo.output.append(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | L Norm (S->T, last batch): {l_norm.item():.2f} | R Norm (T->S, last batch): {r_norm.item():.2f}"
        )
    return left_norms, losses, model, right_norms


@app.cell
def _(DEVICE, mo, model, torch, validation_loader):
    # Evaluation
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for seqw, structw, labelsw in validation_loader:
            seqw, structw, labelsw = (
                seqw.to(DEVICE),
                structw.to(DEVICE),
                labelsw.to(DEVICE).unsqueeze(1),
            )
            outputs, _, _ = model(seqw, structw)
            predicted = torch.sigmoid(outputs) > 0.5
            total_samples += labelsw.size(0)
            total_correct += (predicted == labelsw).sum().item()

    accuracy = 100 * total_correct / total_samples
    mo.output.append(f"\nFinal Validation Accuracy: {accuracy:.2f}%")
    return


@app.cell
def _(left_norms, losses, mo, plt, right_norms):
    # --- Visualization ---
    fig = plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(left_norms, label="Path Norm (Seq -> Struct)")
    plt.plot(right_norms, label="Path Norm (Struct -> Seq)")
    plt.title("Translation Symmetry Dominance")
    plt.xlabel("Epoch")
    plt.ylabel("Pathway Norm")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True)

    plt.tight_layout()
    mo.output.append(fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 📊 Results:
    - ~90% validation accuracy in distinguishing matches from mismatches
    - Consistent norms for both translation directions across training epochs
    - Indicates strong bi-directional consistency between representations

    # 🧠 Conclusions:
    - Sequence and 3Di structure embeddings are nearly symmetric
    - Confirms AlphaFold’s structural encodings preserve information from the original sequence
    - Opens the door to using symmetry deviations as signals of disorder, mutation effects, or model uncertainty

    ## Further Questions
    - Could there be a small amount of asymmetry in the AlphaFold classification?
    - 3Di Compression	The 3Di structural encoding compresses 3D context into discrete tokens → some info loss is expected, especially about long-range contacts, could ChiralNet detect this?
    """
    )
    return


if __name__ == "__main__":
    app.run()
