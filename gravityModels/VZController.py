import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gravityGeodesics import SemanticWavefunction, get_embedding

# Rich console for beautiful logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from textblob import TextBlob
from transformers import GPT2LMHeadModel, GPT2Tokenizer

warnings.filterwarnings("ignore")
console = Console()

class VZController(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),  # ΔRe(z), ΔIm(z)
        )

    def forward(self, state):  # [Re(z), Im(z), V(z), V_target]
        return self.net(state)

    def run_VZController(self, wave_fn, target_curve, steps=500):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        z = torch.tensor([0.5, 0.5], requires_grad=True)

        for t in range(steps):
            v_now = wave_fn.cpt_asymmetry(complex(z[0].item(), z[1].item()))
            v_target = target_curve(t)  # e.g., sinusoidal, step, or user-defined

            state = torch.tensor(
                [z[0].item(), z[1].item(), v_now, v_target], dtype=torch.float32
            )
            dz = self(state)
            z_new = torch.clamp(z + dz, 0, 1)

            v_next = wave_fn.cpt_asymmetry(complex(z_new[0].item(), z_new[1].item()))
            reward = -abs(v_next - v_target)

            loss = -reward
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Fix: Add retain_graph=True
            optimizer.step()

            z = z_new.detach().requires_grad_(
                True
            )  # Fix: Properly detach and re-enable gradients


class SemanticModulationAgent(nn.Module):
    def __init__(self, wave_fn, hidden_dim=64, lr=1e-3):
        super().__init__()
        # Enhanced policy network with memory
        self.policy = nn.GRUCell(4, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim
        self.reset_hidden_state()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.wave_fn = wave_fn
        self.z_history = []
        self.v_history = []
        self.target_history = []
        self.loss_history = []

    def cpt_energy(self, z):
        return self.wave_fn.cpt_asymmetry(complex(z[0].item(), z[1].item()))

    def reset_hidden_state(self):
        """Reset the hidden state for the GRU."""
        self.hidden_state = torch.zeros(1, self.hidden_dim)

    def step(self, z, t, target_fn, clip_bounds=(0, 1)):
        """Take one training step to move toward V_target(t)."""
        v_now = self.cpt_energy(z)
        v_target = target_fn(t)

        state = torch.tensor(
            [z[0].item(), z[1].item(), v_now, v_target], dtype=torch.float32
        )
        
        # Update hidden state with proper gradient handling
        self.hidden_state = self.policy(state.unsqueeze(0), self.hidden_state.detach())
        dz = self.linear(self.hidden_state.squeeze(0))

        # Basic exploration with small random noise
        if np.random.rand() < 0.1:
            exploration_noise = torch.randn_like(dz) * 0.02
            dz = dz + exploration_noise

        z_new = torch.clamp(z + dz, *clip_bounds)
        v_next = self.cpt_energy(z_new)
        reward = torch.tensor(-abs(v_next - v_target), requires_grad=True)

        loss = -reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log
        self.z_history.append(z_new.detach().numpy())
        self.v_history.append(v_next)
        self.target_history.append(v_target)
        self.loss_history.append(loss.item())

        return z_new.detach().requires_grad_(
            True
        ), reward.item()  # Fix: Properly handle gradients

    def train_loop(self, z_init=None, steps=100, target_fn=None):
        if z_init is None:
            z_init = torch.tensor([0.5, 0.5], requires_grad=True)
        z = z_init
        
        # Reset hidden state at start of training
        self.reset_hidden_state()

        # Create rich progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("🧠 RL Training", total=steps)

            for t in range(steps):
                z, reward = self.step(z, t, target_fn)

                # Update progress with current metrics
                if (
                    hasattr(self, "detailed_metrics")
                    and self.detailed_metrics["qualities"]
                ):
                    current_quality = self.detailed_metrics["qualities"][-1]
                    current_sentiment = self.detailed_metrics["sentiments"][-1]
                    progress.update(
                        task,
                        advance=1,
                        description=f"🧠 RL Training [Q:{current_quality:.2f} S:{current_sentiment:.2f}]",
                    )
                else:
                    progress.update(task, advance=1)

                # Show detailed progress every 10 steps
                if t % 10 == 0 and t > 0:
                    self._show_training_status(t, steps)

    def plot_results(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.v_history, label="V(z)")
        plt.plot(self.target_history, "--", label="Target V(t)")
        plt.title("Semantic Energy Tracking")
        plt.xlabel("Step")
        plt.ylabel("CPT Energy")
        plt.legend()

        plt.subplot(1, 2, 2)
        z_hist = np.array(self.z_history)
        plt.plot(z_hist[:, 0], z_hist[:, 1], "o-", color="purple")
        plt.title("Trajectory in Complex Plane")
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def _show_training_status(self, current_step, total_steps):
        """Display detailed training status using rich console."""
        if not hasattr(self, "detailed_metrics"):
            return

        # Get recent metrics (last 10 steps)
        recent_window = min(10, len(self.detailed_metrics["qualities"]))
        if recent_window == 0:
            return

        recent_quality = self.detailed_metrics["qualities"][-recent_window:]
        recent_sentiment = self.detailed_metrics["sentiments"][-recent_window:]
        recent_cpt = self.detailed_metrics["cpt_rewards"][-recent_window:]
        recent_text = self.decoded_texts[-1] if self.decoded_texts else "No text"

        # Create metrics table
        table = Table(title=f"🎯 Training Progress - Step {current_step}/{total_steps}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Current", style="magenta")
        table.add_column("Avg (last 10)", style="green")
        table.add_column("Trend", style="yellow")

        # Quality metrics
        quality_trend = (
            "📈"
            if len(recent_quality) > 5
            and recent_quality[-1] > np.mean(recent_quality[:-1])
            else "📉"
        )
        table.add_row(
            "Text Quality",
            f"{recent_quality[-1]:.3f}",
            f"{np.mean(recent_quality):.3f}",
            quality_trend,
        )

        # Sentiment metrics
        sentiment_trend = (
            "😊"
            if recent_sentiment[-1] > 0
            else "😐"
            if recent_sentiment[-1] == 0
            else "😔"
        )
        table.add_row(
            "Sentiment",
            f"{recent_sentiment[-1]:.3f}",
            f"{np.mean(recent_sentiment):.3f}",
            sentiment_trend,
        )

        # CPT reward
        cpt_trend = "⚡" if recent_cpt[-1] > np.mean(recent_cpt[:-1]) else "⚡"
        table.add_row(
            "CPT Reward",
            f"{recent_cpt[-1]:.3f}",
            f"{np.mean(recent_cpt):.3f}",
            cpt_trend,
        )

        console.print(table)

        # Show latest generated text
        text_panel = Panel(
            recent_text[:150] + "..." if len(recent_text) > 150 else recent_text,
            title="📝 Latest Generated Text",
            border_style="blue",
        )
        console.print(text_panel)


class SemanticModulationAgentWithDecoder(SemanticModulationAgent):
    def __init__(self, wave_fn, hidden_dim=64, lr=1e-3):
        super().__init__(wave_fn, hidden_dim, lr)

        # Load GPT-2 for decoding with proper error handling
        try:
            print("Loading GPT-2 model...")
            self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
            self.decoder.eval()
            self.decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # Fix: Add pad token properly to avoid attention mask warnings
            self.decoder_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.995
            )

            # Create BERT-to-GPT2 mapping layer
            self.bert_to_gpt2_mapper = nn.Linear(768, self.decoder.config.vocab_size)
            print("GPT-2 model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load GPT-2 model: {e}")
            print("Running without text decoder...")
            self.decoder = None
            self.decoder_tokenizer = None
            self.bert_to_gpt2_mapper = None

        self.decoded_texts = []

    def decode_text_from_embedding(self, psi_vec, max_length=20):
        # Check if decoder is available
        if self.decoder is None or self.decoder_tokenizer is None:
            return "No decoder available"

        try:
            # Map the 768-dim BERT embedding to GPT-2 prompt using semantic similarity
            context = self._bert_to_gpt2_prompt(psi_vec)
            encoded = self.decoder_tokenizer(context, return_tensors="pt", padding=True)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask

            with torch.no_grad():
                output = self.decoder.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=self.decoder_tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                )
            return self.decoder_tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            return f"Decode error: {str(e)[:50]}..."

    def _bert_to_gpt2_prompt(self, bert_embedding):
        """Map BERT embedding to meaningful GPT-2 prompt using learned vocabulary mapping."""
        try:
            # Method 1: Use learned linear mapping to find most relevant tokens
            if self.bert_to_gpt2_mapper is not None:
                return self._embedding_to_tokens_prompt(bert_embedding)

            # Method 2: Fallback to statistical analysis
            return self._statistical_prompt_mapping(bert_embedding)

        except Exception as e:
            # Fallback to simple prompt if mapping fails
            return "The situation is interesting and"

    def _embedding_to_tokens_prompt(self, bert_embedding):
        """Use semantic similarity to find meaningful words for prompts."""
        try:
            # Instead of random mapping, use semantic similarity to known concepts
            concept_embeddings = {
                "creativity": get_embedding(
                    "creative imagination inspiration"
                ).detach(),
                "nature": get_embedding("natural environment ecosystem").detach(),
                "technology": get_embedding("digital innovation progress").detach(),
                "emotion": get_embedding("feeling sentiment mood").detach(),
                "knowledge": get_embedding("learning wisdom understanding").detach(),
                "adventure": get_embedding("exploration journey discovery").detach(),
                "harmony": get_embedding("balance peace cooperation").detach(),
                "future": get_embedding("tomorrow progress evolution").detach(),
            }

            # Convert BERT embedding to tensor if needed
            if not torch.is_tensor(bert_embedding):
                bert_embedding = torch.tensor(bert_embedding, dtype=torch.float32)

            # Find most similar concepts using cosine similarity
            similarities = {}
            for concept, concept_emb in concept_embeddings.items():
                sim = torch.nn.functional.cosine_similarity(
                    bert_embedding.unsqueeze(0), concept_emb.unsqueeze(0)
                ).item()
                similarities[concept] = sim

            # Get top 3 most similar concepts
            top_concepts = sorted(
                similarities.items(), key=lambda x: x[1], reverse=True
            )[:3]
            concept_names = [concept for concept, _ in top_concepts]

            # Create more meaningful prompts based on semantic similarity
            prompt_templates = [
                f"The {concept_names[0]} is a {concept_names[1]}",
                f"The {concept_names[0]} is a parent of {concept_names[1]}",
                f"In the realm of {concept_names[0]}, where {concept_names[1]} meets {concept_names[2]},",
                f"Consider how {concept_names[0]} influences {concept_names[1]} through",
                f"The intersection of {concept_names[0]} and {concept_names[1]} reveals",
            ]

            # Select based on embedding characteristics
            emb_hash = abs(hash(str(bert_embedding[:3].tolist()))) % len(
                prompt_templates
            )
            return prompt_templates[emb_hash]

        except Exception as e:
            return self._statistical_prompt_mapping(bert_embedding)

    def _statistical_prompt_mapping(self, bert_embedding):
        """Fallback method using statistical properties of embedding."""
        try:
            # Convert to numpy for easier manipulation
            if torch.is_tensor(bert_embedding):
                emb_np = bert_embedding.detach().cpu().numpy()
            else:
                emb_np = np.array(bert_embedding)

            # Calculate embedding statistics
            mean_val = np.mean(emb_np)
            std_val = np.std(emb_np)
            max_val = np.max(emb_np)
            min_val = np.min(emb_np)

            # Map statistical properties to semantic concepts
            if mean_val > 0.1:
                sentiment_word = "positive"
            elif mean_val < -0.1:
                sentiment_word = "negative"
            else:
                sentiment_word = "neutral"

            if std_val > 0.8:
                complexity_word = "complex"
            elif std_val > 0.5:
                complexity_word = "nuanced"
            else:
                complexity_word = "simple"

            if max_val > 2.0:
                intensity_word = "intense"
            elif max_val > 1.0:
                intensity_word = "moderate"
            else:
                intensity_word = "subtle"

            # Create contextual prompt based on embedding properties
            prompt_templates = [
                f"This {intensity_word} and {complexity_word} situation feels {sentiment_word}.",
                f"In a {complexity_word} way, this {intensity_word} moment is {sentiment_word}.",
                f"The {sentiment_word} nature of this {complexity_word} experience is {intensity_word}.",
                f"Consider this {intensity_word} scenario that is {sentiment_word} and {complexity_word}.",
            ]

            # Select prompt based on embedding hash for consistency
            prompt_idx = int(abs(hash(str(emb_np[:5]))) % len(prompt_templates))
            selected_prompt = prompt_templates[prompt_idx]

            return selected_prompt

        except Exception as e:
            return "The situation is interesting and"

    def sentiment_score(self, text):
        try:
            return TextBlob(text).sentiment.polarity  # Range: [-1, 1]
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0  # Neutral sentiment if error

    def text_quality_score(self, text):
        """Evaluate text quality based on multiple metrics."""
        try:
            if not text or len(text.strip()) < 10:
                return -1.0  # Penalty for very short text

            words = text.split()
            if len(words) < 3:
                return -0.5  # Penalty for too few words

            # Check for repetitive patterns
            unique_words = set(words)
            diversity_ratio = len(unique_words) / len(words)

            # Check for coherence (basic heuristics)
            coherence_score = 0.0
            if any(
                word in text.lower() for word in ["the", "and", "or", "but", "because"]
            ):
                coherence_score += 0.2  # Basic connectors

            if any(char in text for char in ".!?"):
                coherence_score += 0.3  # Proper punctuation

            # Length bonus (but not too long)
            length_score = min(len(words) / 20.0, 1.0)  # Optimal around 20 words

            # Final quality score
            quality = diversity_ratio * 0.4 + coherence_score * 0.4 + length_score * 0.2
            return min(quality, 1.0)

        except Exception as e:
            return 0.0

    def step(self, z, t, target_fn, clip_bounds=(0, 1)):
        """Take one training step with reward = sentiment + CPT target tracking."""
        v_now = self.cpt_energy(z)
        v_target = target_fn(t)

        state = torch.tensor(
            [z[0].item(), z[1].item(), v_now, v_target], dtype=torch.float32
        )
        
        # Update hidden state with proper gradient handling
        self.hidden_state = self.policy(state.unsqueeze(0), self.hidden_state.detach())
        dz = self.linear(self.hidden_state.squeeze(0))

        # Basic exploration with small random noise
        if np.random.rand() < 0.1:
            exploration_noise = torch.randn_like(dz) * 0.02
            dz = dz + exploration_noise
            
        z_new = torch.clamp(z + dz, *clip_bounds)

        v_next = self.cpt_energy(z_new)

        # Get the wavefunction value at the new position
        try:
            psi_complex = self.wave_fn.psi(complex(z_new[0].item(), z_new[1].item()))
            # Handle both complex and real tensors
            if torch.is_complex(psi_complex):
                psi_vec = psi_complex.real
            else:
                psi_vec = psi_complex
            decoded_text = self.decode_text_from_embedding(psi_vec)
        except Exception as e:
            print(f"Wavefunction evaluation error: {e}")
            decoded_text = "Error in wavefunction evaluation"

        sentiment = self.sentiment_score(decoded_text)
        quality = self.text_quality_score(decoded_text)

        # Improved reward function with multiple components
        cpt_reward = -abs(v_next - v_target)  # CPT asymmetry tracking
        text_reward = 0.3 * sentiment + 0.4 * quality  # Text quality & sentiment
        exploration_bonus = 0.1 * np.random.normal(0, 0.05)  # Small exploration noise
        
        # Movement bonus encourages exploration but not too much
        if torch.is_tensor(dz):
            movement_penalty = -0.01 * torch.norm(dz, p=2).item()  # Small penalty for large moves
        else:
            movement_penalty = 0
            
        total_reward = cpt_reward + text_reward + exploration_bonus + movement_penalty
        reward = torch.tensor(total_reward, dtype=torch.float32, requires_grad=True)
        
        loss = -reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Step scheduler if it exists
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        # Enhanced logging
        self.z_history.append(z_new.detach().numpy())
        self.v_history.append(v_next)
        self.target_history.append(v_target)
        self.loss_history.append(loss.item())
        self.decoded_texts.append(decoded_text)

        # Log detailed metrics for analysis
        if not hasattr(self, "detailed_metrics"):
            self.detailed_metrics = {
                "cpt_rewards": [],
                "text_rewards": [],
                "sentiments": [],
                "qualities": [],
            }

        self.detailed_metrics["cpt_rewards"].append(cpt_reward)
        self.detailed_metrics["text_rewards"].append(text_reward)
        self.detailed_metrics["sentiments"].append(sentiment)
        self.detailed_metrics["qualities"].append(quality)

        return z_new.detach().requires_grad_(
            True
        ), reward.item()  # Fix: Properly handle gradients

    def plot_results_with_text(self):
        self.plot_results()

        # Show final training summary with rich formatting
        console.print("\n" + "=" * 80, style="bold blue")
        console.print(
            "🎉 TRAINING COMPLETE! Final Results Summary",
            style="bold green",
            justify="center",
        )
        console.print("=" * 80 + "\n", style="bold blue")

        # Training metrics summary table
        if hasattr(self, "detailed_metrics"):
            summary_table = Table(
                title="📊 Final Training Metrics",
                show_header=True,
                header_style="bold cyan",
            )
            summary_table.add_column("Metric", style="cyan", width=20)
            summary_table.add_column("Final Value", style="magenta", width=15)
            summary_table.add_column("Average (Last 20)", style="green", width=20)
            summary_table.add_column("Improvement", style="yellow", width=15)

            # Handle large values with scientific notation and scaling
            def format_large_value(value, decimals=3):
                if abs(value) > 1000:
                    return f"{value:.2e}"
                return f"{value:.{decimals}f}"

            final_quality = (
                self.detailed_metrics["qualities"][-1]
                if self.detailed_metrics["qualities"]
                else 0
            )
            avg_quality = (
                np.mean(self.detailed_metrics["qualities"][-20:])
                if self.detailed_metrics["qualities"]
                else 0
            )
            initial_quality = (
                np.mean(self.detailed_metrics["qualities"][:5])
                if len(self.detailed_metrics["qualities"]) > 5
                else final_quality
            )
            quality_improvement = (
                (final_quality - initial_quality) / max(initial_quality, 0.001)
            ) * 100

            final_sentiment = (
                self.detailed_metrics["sentiments"][-1]
                if self.detailed_metrics["sentiments"]
                else 0
            )
            avg_sentiment = (
                np.mean(self.detailed_metrics["sentiments"][-20:])
                if self.detailed_metrics["sentiments"]
                else 0
            )

            final_cpt = (
                self.detailed_metrics["cpt_rewards"][-1]
                if self.detailed_metrics["cpt_rewards"]
                else 0
            )
            avg_cpt = (
                np.mean(self.detailed_metrics["cpt_rewards"][-20:])
                if self.detailed_metrics["cpt_rewards"]
                else 0
            )

            summary_table.add_row(
                "Text Quality",
                format_large_value(final_quality),
                format_large_value(avg_quality),
                f"{quality_improvement:+.1f}%",
            )
            summary_table.add_row(
                "Sentiment",
                format_large_value(final_sentiment),
                format_large_value(avg_sentiment),
                "😊" if avg_sentiment > 0 else "😐",
            )
            summary_table.add_row(
                "CPT Reward",
                format_large_value(final_cpt),
                format_large_value(avg_cpt),
                "⚡ Active",
            )

            console.print(summary_table)

        console.print(
            "\n📝 Latest Generated Texts (Best Results):", style="bold yellow"
        )

        # Show recent texts in a nice format
        recent_texts = self.decoded_texts[-10:]
        for i, text in enumerate(recent_texts):
            step_num = len(self.decoded_texts) - 10 + i
            quality = (
                self.text_quality_score(text)
                if hasattr(self, "text_quality_score")
                else 0
            )
            sentiment = (
                self.sentiment_score(text) if hasattr(self, "sentiment_score") else 0
            )

            # Color code based on quality
            quality_color = (
                "green" if quality > 0.7 else "yellow" if quality > 0.5 else "red"
            )
            sentiment_emoji = (
                "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😔"
            )

            # Create text panel with formatted metrics
            text_content = text[:120] + "..." if len(text) > 120 else text
            
            # Format values properly
            def format_value(value, decimals=2):
                if abs(value) > 1000:
                    return f"{value:.2e}"
                return f"{value:.{decimals}f}"
            
            panel = Panel(
                text_content,
                title=f"Step {step_num} | Q:{format_value(quality, 2)} S:{format_value(sentiment, 2)} {sentiment_emoji}",
                border_style=quality_color,
                width=80,
            )
            console.print(panel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    try:
        print("Initializing semantic embeddings...")
        x0 = get_embedding("If I were a cat, I would be a tiger").detach()
        x1 = get_embedding("If I were a cat, I would be a dog").detach()
        wave = SemanticWavefunction(x0, x1)
        print("Semantic wavefunction created successfully!")

        def target_curve(t):
            # More meaningful target that encourages semantic exploration
            # Start low (similar concepts), gradually increase (more diverse concepts)
            base_diversity = 2.0
            exploration_amplitude = 3.0
            exploration_frequency = 0.1

            # Add some structured variation to encourage different semantic regions
            semantic_target = base_diversity + exploration_amplitude * np.sin(
                t * exploration_frequency
            )

            # Add curriculum learning: start easier, get harder
            curriculum_factor = min(
                t / 50.0, 1.0
            )  # Ramp up difficulty over first 50 steps

            return semantic_target * (0.5 + 0.5 * curriculum_factor)

        # Beautiful startup banner
        console.print("=" * 80, style="bold cyan")
        console.print(
            "🚀 SEMANTIC RL TRAINING INITIALIZED", style="bold green", justify="center"
        )
        console.print(
            "Navigating Complex Semantic Space with CPT Symmetry",
            style="italic blue",
            justify="center",
        )
        console.print("=" * 80, style="bold cyan")

        startup_table = Table(show_header=False, box=None)
        startup_table.add_column("", style="cyan", width=25)
        startup_table.add_column("", style="white", width=30)
        startup_table.add_row("🧠 Semantic Embeddings:", "Cat -> Tiger vs Cat -> Dog")
        startup_table.add_row("🎯 Target Function:", "Curriculum Learning Enabled")
        startup_table.add_row("📊 Tracking Metrics:", "Quality, Sentiment, CPT Reward")
        startup_table.add_row("🔄 Training Steps:", f"{args.steps} iterations")
        console.print(startup_table)
        console.print()

        # Test agent with decoder
        console.print(
            "🤖 Initializing Semantic Modulation Agent...", style="bold yellow"
        )
        agent = SemanticModulationAgentWithDecoder(wave_fn=wave)

        console.print("🎓 Starting RL Training Loop...", style="bold green")
        agent.train_loop(steps=args.steps, target_fn=target_curve)

        agent.plot_results_with_text()
        console.print("\n✅ Agent training completed successfully!", style="bold green")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback

        traceback.print_exc() 