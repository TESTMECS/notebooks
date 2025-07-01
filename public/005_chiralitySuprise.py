# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets==3.6.0",
#     "marimo",
#     "matplotlib==3.10.3",
#     "nltk==3.9.1",
#     "numpy==2.3.1",
#     "torch==2.7.1",
#     "transformers==4.53.0",
# ]
# ///

import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # The Experimental Design: The Semantic Compressor

        **The Goal**: Train a network to reconstruct a sentence from two views: the full sentence, and a "skeleton" version where all the predictable words are masked out.

        **The "Redundancy" Proxy**: We'll define "predictable" words as common stopwords ("the", "a", "is", "on", etc.). These words form the grammatical scaffolding of the language but carry little unique information.

    **The ChiralNet Autoencoder:**: An autoencoder. It takes sequences in and must output sequences of the same shape. The core will be two CNN-based pathways. The left pathway sees the original sentence. The right pathway sees the masked sentence (the "skeleton"). 

    **The "Surprise" Vector**: The `l_out - r_out` difference vector should, in theory, represent the information content of the unpredictable words that were masked out.

    **The Decoder**: A separate module takes this "surprise" vector and attempts to reconstruct the original, complete sentence.

    **The Training Task**: The loss function will measure how accurately the decoder's output matches the original input sentence.

    **The Groundbreaking Hypothesis**: To minimize reconstruction loss, the network must learn to decompose the sentence.

    **The Master Pathway (left) **learns to create a rich feature representation of the entire, correct sentence.

    **The Assistant Pathway (right)** learns the features of the grammatical skeleton.

    The **Decoder** learns to take the "surprise" vector—the difference between the two—and use it to fill in the blanks of the skeleton.

    Effectively, the **Assistant Pathway** has been forced to learn a model of pure information or novelty.
    """
    )
    return


@app.cell
def _():
    # --- The ChiralNet "Surprise Extractor" ---
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer
    import nltk
    import gc  # For memory cleanup

    return (
        AutoTokenizer,
        DataLoader,
        Dataset,
        gc,
        load_dataset,
        nltk,
        nn,
        np,
        plt,
        torch,
    )


@app.cell
def _(nltk):
    # --- Download NLTK stopwords ---
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    return (stopwords,)


@app.cell
def _(torch):
    # --- Configuration ---
    torch.manual_seed(42)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 500  # Autoencoding complex data takes time
    BATCH_SIZE = 32  # Can increase now that model is tiny
    ACCUMULATION_STEPS = 2  # Smaller accumulation since batch is larger
    LEARNING_RATE = 0.003  # Slightly higher for smaller model
    MAX_LENGTH = 16  # Even smaller sequences for memory efficiency
    SUBSET_SIZE = 2000  # Smaller dataset for quick experimentation
    VOCAB_SIZE = 2000  # Increased from 1000 - let's test bigger vocab!
    USE_MIXED_PRECISION = torch.cuda.is_available()  # Enable for GPU
    return (
        ACCUMULATION_STEPS,
        BATCH_SIZE,
        DEVICE,
        EPOCHS,
        LEARNING_RATE,
        MAX_LENGTH,
        SUBSET_SIZE,
        USE_MIXED_PRECISION,
        VOCAB_SIZE,
    )


@app.cell
def _(DEVICE, torch):
    print(f"--- ChiralNet Surprise Extractor ---")
    print(f"Using device: {DEVICE}")

    # Memory reporting for CUDA
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory Available: {gpu_memory:.1f} GB")
        torch.cuda.empty_cache()  # Start with clean memory

    return


@app.cell
def _(Dataset, load_dataset, mo, stopwords, torch):
    # --- Custom Dataset for Masking ---
    class SurpriseExtractorDataset(Dataset):
        def __init__(
            self, tokenizer, max_length, subset_size, vocab_size, split="train"
        ):
            self.tokenizer = tokenizer
            self.max_length = max_length

            print("Loading dataset and preparing masks...")
            dataset = load_dataset("ag_news", split=split).select(range(subset_size))
            # We only need the text, not the labels
            self.texts = [item["text"] for item in dataset]

            # Get English stopwords and tokenize them (optimized)
            stop_words = set(stopwords.words("english"))
            # Pre-filter stopwords to only include those in vocab
            self.stopword_ids = set()
            for sw in stop_words:
                try:
                    if sw in tokenizer.vocab:
                        self.stopword_ids.add(tokenizer.vocab[sw])
                except:
                    continue  # Skip if tokenization fails

            # Create a smart vocabulary mapping using dataset frequency
            self.setup_smart_vocab_mapping(tokenizer, vocab_size)

        def setup_smart_vocab_mapping(self, tokenizer, target_vocab_size):
            """Create vocabulary using actual dataset word frequency - much smarter!"""
            from collections import Counter
            import re

            print(
                f"🧠 Analyzing dataset to find {target_vocab_size} most important words..."
            )

            # Step 1: Tokenize all texts and count word frequencies
            all_words = []
            for text in self.texts:
                # Clean and split text into words
                words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
                all_words.extend(words)

            # Step 2: Count word frequencies
            word_counts = Counter(all_words)

            # Step 3: Get most frequent words that exist in tokenizer vocab
            valid_words = []
            for word, count in word_counts.most_common(
                target_vocab_size * 3
            ):  # Get extra to filter
                if word in tokenizer.vocab and len(word) > 1:  # Skip single chars
                    valid_words.append((word, count))
                if len(valid_words) >= target_vocab_size:
                    break

            # Step 4: Add some essential words if missing
            essential_words = [
                "the",
                "and",
                "to",
                "of",
                "a",
                "in",
                "is",
                "it",
                "you",
                "that",
                "he",
                "was",
                "for",
                "on",
                "are",
                "as",
                "with",
                "his",
                "they",
                "i",
                "at",
                "be",
                "this",
                "have",
                "from",
                "or",
                "one",
                "had",
                "by",
                "word",
                "but",
                "not",
                "what",
                "all",
                "were",
                "we",
                "when",
            ]

            existing_words = {word for word, _ in valid_words}
            for essential in essential_words:
                if essential not in existing_words and essential in tokenizer.vocab:
                    valid_words.append((essential, 1000))  # High priority
                    if len(valid_words) >= target_vocab_size:
                        break

            # Step 5: Create the mapping
            self.small_vocab_ids = []
            self.id_to_word = {}
            self.word_to_small_id = {}

            for i, (word, count) in enumerate(valid_words[:target_vocab_size]):
                token_id = tokenizer.vocab[word]
                self.small_vocab_ids.append(token_id)
                self.id_to_word[i] = token_id
                self.word_to_small_id[word] = i

            # Step 6: Create mapping from any token ID to our small vocab
            self.vocab_mapping = {}
            for i in range(tokenizer.vocab_size):
                self.vocab_mapping[i] = i % len(self.small_vocab_ids)

            mo.md(
                f"✅ Smart vocabulary created! Top words: {[word for word, _ in valid_words[:10]]}"
            )
            mo.md(
                f"📊 Frequency range: {valid_words[0][1]} (most) to {valid_words[-1][1]} (least)"
            )

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            # Tokenize the original sentence
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            original_ids = encoding["input_ids"].squeeze(0)

            # Create the masked "skeleton" version (optimized)
            masked_ids = original_ids.clone()
            mask = torch.isin(original_ids, torch.tensor(list(self.stopword_ids)))
            masked_ids[mask] = self.tokenizer.mask_token_id

            # Map to small vocabulary
            original_small = torch.tensor(
                [self.vocab_mapping[id.item()] for id in original_ids]
            )
            masked_small = torch.tensor(
                [self.vocab_mapping[id.item()] for id in masked_ids]
            )

            return original_small, masked_small

    return (SurpriseExtractorDataset,)


@app.cell
def _(MAX_LENGTH, nn):
    # --- Model Definition (TINY VERSION - No More Billion-Parameter Monsters!) ---
    class ChiralCompressorNet(nn.Module):
        def __init__(
            self, small_vocab_size, embed_dim=16, feature_dim=32
        ):  # TINY dimensions
            super().__init__()

            self.small_vocab_size = small_vocab_size

            # Tiny embedding
            self.embedding = nn.Embedding(small_vocab_size, embed_dim)

            # Simple encoder - just a few linear layers
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim * MAX_LENGTH, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
            )

            # Tiny decoder that outputs to small vocab
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim // 2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, MAX_LENGTH * small_vocab_size),
            )

        def forward(self, x_original, x_masked):
            batch_size = x_original.size(0)

            # Embed and flatten for simple linear layers
            original_embed = self.embedding(x_original).view(batch_size, -1)
            masked_embed = self.embedding(x_masked).view(batch_size, -1)

            # Left pathway sees the full sentence
            l_out = self.encoder(original_embed)
            # Right pathway sees the grammatical skeleton
            r_out = self.encoder(masked_embed)

            # The "surprise" vector is their difference
            surprise_vector = l_out - r_out

            # Decode to small vocabulary
            reconstructed_logits = self.decoder(surprise_vector)
            reconstructed_logits = reconstructed_logits.view(
                batch_size, self.small_vocab_size, MAX_LENGTH
            )

            return reconstructed_logits, l_out.norm(), r_out.norm()

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters())

    return (ChiralCompressorNet,)


@app.cell
def _(
    AutoTokenizer,
    BATCH_SIZE,
    DataLoader,
    MAX_LENGTH,
    SUBSET_SIZE,
    SurpriseExtractorDataset,
    VOCAB_SIZE,
    torch,
):
    # --- Main Execution Block ---
    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Setup DataLoaders
    train_dataset = SurpriseExtractorDataset(
        tokenizer, MAX_LENGTH, SUBSET_SIZE, VOCAB_SIZE, split="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return tokenizer, train_dataset, train_loader


@app.cell
def _(
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    ChiralCompressorNet,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    USE_MIXED_PRECISION,
    gc,
    mo,
    nn,
    tokenizer,
    torch,
    train_dataset,
    train_loader,
):
    # Setup Model
    model = ChiralCompressorNet(small_vocab_size=len(train_dataset.small_vocab_ids)).to(
        DEVICE
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    # We ignore the padding token in the loss calculation
    loss_fn = nn.CrossEntropyLoss()

    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None

    # --- Training Loop ---
    mo.output.append("\n--- Training the Surprise Extractor ---")
    mo.output.append(
        f"🚀 Performance Mode: Batch Size {BATCH_SIZE}, Gradient Accumulation {ACCUMULATION_STEPS}"
    )
    mo.output.append(f"💾 Mixed Precision: {'Enabled' if USE_MIXED_PRECISION else 'Disabled'}")
    mo.output.append(f"📊 Total Parameters: {model.count_parameters():,} (MUCH better!)")
    mo.output.append(
        f"🎯 Vocab Size: {tokenizer.vocab_size:,} -> {len(train_dataset.small_vocab_ids):,}"
    )

    left_norms, right_norms, losses = [], [], []
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        step_count = 0

        for i, (original_batch, masked_batch) in enumerate(train_loader):
            original_batch, masked_batch = (
                original_batch.to(DEVICE, non_blocking=True),
                masked_batch.to(DEVICE, non_blocking=True),
            )

            # Mixed precision forward pass (fixed deprecation warning)
            if USE_MIXED_PRECISION:
                with torch.amp.autocast("cuda"):
                    recon_logits, l_norm, r_norm = model(original_batch, masked_batch)
                    # Target is already in small vocab space
                    loss = loss_fn(recon_logits, original_batch) / ACCUMULATION_STEPS
            else:
                recon_logits, l_norm, r_norm = model(original_batch, masked_batch)
                loss = loss_fn(recon_logits, original_batch) / ACCUMULATION_STEPS

            # Backward pass with gradient accumulation
            if USE_MIXED_PRECISION:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * ACCUMULATION_STEPS

            # Update weights every ACCUMULATION_STEPS
            if (i + 1) % ACCUMULATION_STEPS == 0:
                if USE_MIXED_PRECISION:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                step_count += 1

                # Memory cleanup every few steps
                if step_count % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        left_norms.append(l_norm.item())
        right_norms.append(r_norm.item())
        mo.output.append(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | L Norm: {l_norm.item():.2f} | R Norm: {r_norm.item():.2f}"
        )

    # --- Verification and Demonstration ---
    model.eval()
    with torch.no_grad():
        # Grab one batch from the dataloader to test
        original_sample, masked_sample = next(iter(train_loader))
        original_sample, masked_sample = (
            original_sample.to(DEVICE),
            masked_sample.to(DEVICE),
        )

        # Get the model's reconstruction
        if USE_MIXED_PRECISION:
            with torch.amp.autocast("cuda"):
                recon_logits, _, _ = model(original_sample, masked_sample)
        else:
            recon_logits, _, _ = model(original_sample, masked_sample)

        recon_ids = torch.argmax(recon_logits, dim=1)

        # Decode and print the first 3 examples
        # Now we can decode to actual words!
        for i in range(min(3, original_sample.size(0))):
            # Convert back to actual token IDs for decoding
            original_tokens = [
                train_dataset.id_to_word[idx.item()] for idx in original_sample[i]
            ]
            masked_tokens = [
                train_dataset.id_to_word[idx.item()] for idx in masked_sample[i]
            ]
            recon_tokens = [
                train_dataset.id_to_word[idx.item()] for idx in recon_ids[i]
            ]

            # Decode to text
            original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
            masked_text = tokenizer.decode(masked_tokens, skip_special_tokens=True)
            recon_text = tokenizer.decode(recon_tokens, skip_special_tokens=True)

            mo.output.append("-" * 50)
            mo.output.append(f"ORIGINAL:  {original_text}")
            mo.output.append(f"MASKED:    {masked_text}")
            mo.output.append(f"RECONSTRUCTED: {recon_text}")
            mo.output.append(
                f"✨ Now showing real words from {len(train_dataset.small_vocab_ids)}-word vocabulary!"
            )

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    mo.output.append(
        f"\n🎉 Training completed! Model size: {model.count_parameters():,} parameters"
    )
    return left_norms, losses, right_norms


@app.cell
def _(left_norms, losses, mo, np, plt, right_norms):
    # --- Visualizing ChiralNet Learning Dynamics ---
    mo.output.append("\n📊 Visualizing ChiralNet Pathway Evolution...")

    epochs = range(1, len(losses) + 1)

    # Create a comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "🧬 ChiralNet Learning Dynamics: Left vs Right Pathways",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Loss over epochs
    ax1.plot(epochs, losses, "b-", linewidth=2, marker="o", markersize=4)
    ax1.set_title("📈 Reconstruction Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    # Plot 2: Left vs Right Norms
    ax2.plot(
        epochs,
        left_norms,
        "r-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Left Pathway (Full Text)",
    )
    ax2.plot(
        epochs,
        right_norms,
        "g-",
        linewidth=2,
        marker="^",
        markersize=4,
        label="Right Pathway (Masked Text)",
    )
    ax2.set_title("🧬 Pathway Norm Evolution", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Norm Magnitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f9fa")

    # Plot 3: Norm Difference (Surprise Signal)
    surprise_signal = np.array(left_norms) - np.array(right_norms)
    ax3.plot(epochs, surprise_signal, "purple", linewidth=2, marker="D", markersize=4)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_title('✨ "Surprise" Signal (L-R Difference)', fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Norm Difference")
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#f8f9fa")

    # Plot 4: Norm Ratio
    norm_ratio = np.array(left_norms) / (
        np.array(right_norms) + 1e-8
    )  # Avoid division by zero
    ax4.plot(epochs, norm_ratio, "orange", linewidth=2, marker="*", markersize=6)
    ax4.axhline(y=1, color="black", linestyle="--", alpha=0.5, label="Equal Norms")
    ax4.set_title("⚖️ Pathway Balance (L/R Ratio)", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Left/Right Ratio")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#f8f9fa")

    plt.tight_layout()
    mo.output.append(fig)

    # Print some insights
    mo.output.append(f"\n🔍 ChiralNet Analysis:")
    mo.output.append(f"   📊 Final Loss: {losses[-1]:.4f}")
    mo.output.append(f"   🔴 Left Pathway (Full): {left_norms[-1]:.2f}")
    mo.output.append(f"   🟢 Right Pathway (Masked): {right_norms[-1]:.2f}")
    mo.output.append(f"   ✨ Final Surprise Signal: {surprise_signal[-1]:.2f}")
    mo.output.append(f"   ⚖️ Final L/R Ratio: {norm_ratio[-1]:.2f}")

    if surprise_signal[-1] > 0:
        mo.output.append(f"   💡 The left pathway (full text) has stronger features - good sign!")
    else:
        mo.output.append(f"   🤔 The right pathway (masked text) is stronger - interesting!")

    mo.output.append(
        f"   📈 Loss improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%"
    )
    return


if __name__ == "__main__":
    app.run()
