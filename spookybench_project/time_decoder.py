"""SpookyBench Time-Aware Decoder
--------------------------------

The SpookyBench dataset is loaded from Hugging Face. Each video sample
contains frames that appear as noise individually but reveal meaningful
content (text, objects, shapes, dynamic scenes) when viewed as temporal

sequences. Frames are processed with a vision transformer and refined
with a learned ``GravityField``. The ``TimeAwareDecoder`` consumes these
causally ordered embeddings.
"""  # noqa: E501

from __future__ import annotations

import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class GravityField(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # ensures phi > 0
        )

    def forward(self, coords):
        """
        coords: Tensor of shape (batch_size, 4) where columns are [t, x, y, z] or equivalent
        returns: Scalar phi > 0 per point
        """  # noqa: E501
        return self.net(coords).squeeze(-1)  # (batch_size,)


def project_until_convergence(
    pairs, spatial, time_vec, phi_model, eps1=1e-5, max_passes=10000
):
    num_passes = 0
    converged = False
    time_vec = time_vec.clone().detach().requires_grad_(False)
    spatial = spatial.clone().detach().requires_grad_(False)

    while not converged and num_passes < max_passes:
        num_passes += 1
        converged = True

        for i, j in pairs:
            t_i, t_j = time_vec[i], time_vec[j]
            x_i, x_j = spatial[i], spatial[j]
            delta_t = t_i - t_j
            delta_x = x_i - x_j
            dx2 = torch.sum(delta_x**2)

            coords_i = torch.cat([t_i.view(1), x_i])
            phi_i = phi_model(coords_i.unsqueeze(0))[0]

            ds2 = -phi_i * delta_t**2 + dx2

            if ds2 >= -eps1:
                new_delta_t = torch.sqrt((dx2 + eps1) / phi_i + 1e-9)
                time_vec[i] = t_j + new_delta_t.item()
                converged = False

    return time_vec


def causal_refine_embeddings_with_phi(
    frame_embeddings, attention_matrix, phi_model, epsilon=1e-5
):
    """
    Args:
        frame_embeddings: Tensor of shape (seq_len, hidden_dim) - embeddings for video frames
        attention_matrix: Tensor of shape (seq_len, seq_len), computed attention between frames
        phi_model: trained GravityField instance
        epsilon: threshold for ds² constraint

    Returns:
        refined_spacetime_coords: (seq_len, 4)
        refined_embeddings: updated frame_embeddings with causal feedback
    """  # noqa: E501
    seq_len, hidden_dim = frame_embeddings.shape
    device = frame_embeddings.device

    # === Step 1: Project into spacetime ===
    spatial = frame_embeddings[
        :, :3
    ]  # take first 3 dims as proxy spatial
    time_vec = torch.zeros(seq_len, device=device)

    # Extract top-k attention edges (causal order)
    k = 2
    pairs = []
    for i in range(seq_len):
        topk_indices = torch.topk(attention_matrix[i], k).indices
        for j in topk_indices:
            if j.item() < i:
                pairs.append((
                    i,
                    j.item(),
                ))  # frame i attends to frame j (past)

    # === Step 2: Project until causal convergence ===
    time_vec = project_until_convergence(
        pairs, spatial, time_vec, phi_model, eps1=epsilon
    )

    # === Step 3: Combine t and x into spacetime coord ===
    spacetime_coords = torch.cat(
        [time_vec.unsqueeze(-1), spatial], dim=-1
    )

    # === Step 4: Feed back into frame embedding layer ===
    refined_embeddings = torch.cat(
        [spacetime_coords, frame_embeddings[:, 3:]], dim=-1
    )  # prepend causal txy

    return spacetime_coords, refined_embeddings


def load_spookybench(split: str = "train[:10]"):
    """Load a small portion of the SpookyBench video dataset."""
    return load_dataset(
        "timeblindness/spooky-bench", name="default", split=split
    )


class TimeAwareDecoder(nn.Module):
    """Decoder that predicts temporal patterns from video frame sequences."""  # noqa: E501

    def __init__(
        self,
        hidden_dim: int = 768,
        hidden_size: int = 256,
        num_classes: int = 4,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + 1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(
            hidden_size // 2, num_classes
        )  # 4 categories: text, objects, scenes, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def extract_frames_from_video_bytes(video_bytes):
    """
    Extract frames from video bytes data using OpenCV.

    Args:
        video_bytes: Raw video data as bytes

    Returns:
        frames: List of numpy arrays representing video frames
    """
    try:


        # Write bytes to temporary file for OpenCV to read
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            print("Failed to open video file")
            os.unlink(temp_path)
            return []

        frames = []
        frame_count = 0
        max_frames = 100  # Limit frames for processing efficiency

        print("Extracting frames from video...")

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"Extracted {frame_count} frames...")

        cap.release()
        os.unlink(temp_path)  # Clean up temporary file

        print(f"Successfully extracted {len(frames)} frames")
        return frames

    except ImportError:
        print(
            "OpenCV (cv2) not installed. Install with: pip install opencv-python"  # noqa: E501
        )
        return []
    except Exception as e:
        print(f"Error processing video bytes: {e}")
        return []


def process_video_sample(processor, model, phi_model, video_bytes):
    """
    Process a video sample from SpookyBench dataset.

    Args:
        processor: Image processor for vision transformer
        model: Vision transformer model
        phi_model: Gravity field model
        video_bytes: Raw video data bytes

    Returns:
        refined_embeddings: Processed frame embeddings with temporal causal structure
    """  # noqa: E501

    # Extract frames from video bytes
    frames = extract_frames_from_video_bytes(video_bytes)

    if not frames:
        print("No frames extracted from video")
        return None

    # Process frames through vision transformer
    frame_embeddings = []
    for frame in frames:
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        # Process frame
        inputs = processor(frame, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Use pooled output or mean of last hidden states
            embedding = outputs.last_hidden_state.mean(
                dim=1
            )  # (1, hidden_dim)
            frame_embeddings.append(embedding)

    if not frame_embeddings:
        return None

    # Stack frame embeddings
    frame_embeddings = torch.cat(
        frame_embeddings, dim=0
    )  # (num_frames, hidden_dim)

    # Compute simple attention matrix between frames (placeholder)
    # In practice, you might use a more sophisticated attention mechanism  # noqa: E501
    seq_len = frame_embeddings.shape[0]
    attention_matrix = torch.softmax(
        torch.randn(seq_len, seq_len), dim=-1
    )  # Placeholder attention

    # Apply causal refinement
    _, refined = causal_refine_embeddings_with_phi(
        frame_embeddings, attention_matrix, phi_model
    )

    return refined


def main():
    # Use a vision transformer instead of BERT for processing images/videos  # noqa: E501
    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )
    vision_model = AutoModel.from_pretrained(
        "google/vit-base-patch16-224"
    )

    phi_model = GravityField(input_dim=4)
    decoder = TimeAwareDecoder(
        hidden_dim=768, hidden_size=256, num_classes=4
    )

    dataset = load_spookybench()

    print(f"Dataset loaded with {len(dataset)} samples")
    print("=" * 60)

    # Store results for validation
    all_predictions = []
    all_ground_truth = []
    all_categories = []

    # Create output directory for visualizations
    output_dir = "validation_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for i, item in enumerate(dataset):
        print(f"\n{'=' * 20} SAMPLE {i + 1} {'=' * 20}")

        # Extract ground truth category
        relative_path = item.get("relative_path", "unknown")
        category_name, category_idx = extract_category_from_path(
            relative_path
        )

        print(f"File: {relative_path}")
        print(
            f"Ground Truth Category: {category_name} (index: {category_idx})"  # noqa: E501
        )

        # SpookyBench items contain video data in bytes format
        if "file" in item and "bytes" in item["file"]:
            video_bytes = item["file"]["bytes"]
            print(f"Video data size: {len(video_bytes)} bytes")

            # Extract frames for analysis
            frames = extract_frames_from_video_bytes(video_bytes)

            if frames:
                # Analyze temporal patterns
                temporal_analysis = analyze_temporal_patterns(frames)
                print("\nTemporal Analysis:")
                print(
                    f"  Motion variance: {temporal_analysis.get('motion_variance', 0):.4f}"  # noqa: E501
                )
                print(
                    f"  Motion mean: {temporal_analysis.get('motion_mean', 0):.4f}"  # noqa: E501
                )
                print(
                    f"  Has significant motion: {temporal_analysis.get('has_significant_motion', False)}"  # noqa: E501
                )

                # Visualize frames
                viz_path = os.path.join(
                    output_dir,
                    f"sample_{i + 1}_{category_name}_frames.png",
                )
                visualize_frames(
                    frames,
                    title=f"Sample {i + 1}: {category_name} - {relative_path.split('/')[-1]}",  # noqa: E501
                    save_path=viz_path,
                )

                # Process the video sample through the model
                refined = process_video_sample(
                    processor, vision_model, phi_model, video_bytes
                )

                if refined is not None:
                    pred = decoder(refined)

                    # Get prediction summary
                    pred_probs = torch.softmax(pred, dim=1)
                    pred_class = torch.argmax(pred_probs, dim=1)
                    confidence = torch.max(pred_probs, dim=1)[0]

                    category_names = [
                        "text",
                        "objects",
                        "dynamic_scenes",
                        "shapes",
                    ]

                    print("\nModel Predictions:")
                    print(f"  Prediction shape: {pred.shape}")
                    print(
                        f"  Average predicted class: {pred_class.float().mean().item():.2f}"  # noqa: E501
                    )
                    print(
                        f"  Average confidence: {confidence.mean().item():.4f}"  # noqa: E501
                    )

                    # Show per-frame predictions
                    print("  Frame-by-frame predictions:")
                    for frame_idx in range(
                        min(5, len(pred_class))
                    ):  # Show first 5 frames
                        predicted_category = category_names[
                            pred_class[frame_idx]
                        ]
                        frame_confidence = confidence[frame_idx]
                        print(
                            f"    Frame {frame_idx}: {predicted_category} (conf: {frame_confidence:.3f})"  # noqa: E501
                        )

                    # Check if model is detecting the right category
                    most_common_pred = torch.mode(pred_class)[
                        0
                    ].item()
                    is_correct = most_common_pred == category_idx
                    print(
                        f"  Most common prediction: {category_names[most_common_pred]}"  # noqa: E501
                    )
                    print(
                        "  ✓ CORRECT"
                        if is_correct
                        else "  ✗ INCORRECT"
                    )

                    # Store aggregated prediction for overall validation (one per video)  # noqa: E501
                    # Use the most common prediction across all frames
                    video_prediction = torch.zeros(
                        1, 4
                    )  # One prediction per video
                    video_prediction[0, most_common_pred] = (
                        1.0  # One-hot encode the most common prediction  # noqa: E501
                    )
                    all_predictions.append(video_prediction)
                    all_ground_truth.append(category_idx)
                    all_categories.append(category_name)

                else:
                    print("Failed to process video sample")
            else:
                print("No frames extracted from video")
        else:
            print(f"Unexpected item structure: {item}")

        # Process only first few samples for testing
        if i >= 4:  # Increased to get more validation data
            break

    # Overall validation results
    if all_predictions and all_ground_truth:
        print(f"\n{'=' * 20} OVERALL VALIDATION {'=' * 20}")

        # Stack all predictions
        all_preds_tensor = torch.cat(all_predictions, dim=0)
        all_gt_array = np.array(all_ground_truth)

        # Validate predictions
        validation_results = validate_predictions(
            all_preds_tensor, all_gt_array, all_categories
        )

        print("Overall Results:")
        print(
            f"  Total samples processed: {validation_results['total_predictions']}"  # noqa: E501
        )
        print(
            f"  Correct predictions: {validation_results['correct_predictions']}"  # noqa: E501
        )
        print(
            f"  Overall accuracy: {validation_results['overall_accuracy']:.2%}"  # noqa: E501
        )

        print("\nPer-category results:")
        for category, stats in validation_results[
            "category_stats"
        ].items():
            print(
                f"  {category}: {stats['accuracy']:.2%} ({stats['count']} samples)"  # noqa: E501
            )

        print(
            "\nNote: This is an untrained model, so low accuracy is expected."  # noqa: E501
        )
        print(
            "The key validation is that temporal patterns are being detected and processed."  # noqa: E501
        )

    print(
        f"\nValidation complete! Check {output_dir}/ for frame visualizations."  # noqa: E501
    )


def extract_category_from_path(relative_path):
    """
    Extract category from SpookyBench file path.
    Categories: text, objects, dynamic_scenes, shapes
    """
    path_lower = relative_path.lower()

    if "words" in path_lower or "text" in path_lower:
        return "text", 0
    elif "objects" in path_lower or "object" in path_lower:
        return "objects", 1
    elif "dynamic_scenes" in path_lower or "scene" in path_lower:
        return "dynamic_scenes", 2
    elif "shapes" in path_lower or "shape" in path_lower:
        return "shapes", 3
    else:
        # Try to infer from path structure
        parts = relative_path.split("/")
        if len(parts) > 0:
            category = parts[0].lower()
            category_map = {
                "words": ("text", 0),
                "text": ("text", 0),
                "objects": ("objects", 1),
                "dynamic_scenes": ("dynamic_scenes", 2),
                "scenes": ("dynamic_scenes", 2),
                "shapes": ("shapes", 3),
            }
            return category_map.get(category, ("unknown", -1))

    return "unknown", -1


def visualize_frames(
    frames, title="Video Frames", max_frames=8, save_path=None
):
    """
    Visualize a subset of video frames to see the temporal patterns.
    """
    if not frames:
        print("No frames to visualize")
        return

    num_frames = min(len(frames), max_frames)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_frames):
        frame = frames[
            i * len(frames) // num_frames
        ]  # Sample frames evenly
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {i * len(frames) // num_frames}")
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        # Save to default location instead of showing
        default_path = f"frames_visualization_{np.random.randint(1000, 9999)}.png"  # noqa: E501
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {default_path}")

    plt.close()


def analyze_temporal_patterns(frames, window_size=5):
    """
    Analyze temporal patterns in the video frames.
    Compute frame differences and motion metrics.
    """
    if len(frames) < 2:
        return {}

    # Convert frames to grayscale for analysis
    gray_frames = []
    for frame in frames:
        if len(frame.shape) == 3:
            # Convert RGB to grayscale
            gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = frame
        gray_frames.append(gray)

    # Compute frame differences
    frame_diffs = []
    for i in range(1, len(gray_frames)):
        diff = np.abs(gray_frames[i] - gray_frames[i - 1])
        frame_diffs.append(np.mean(diff))

    # Compute motion metrics
    motion_variance = np.var(frame_diffs)
    motion_mean = np.mean(frame_diffs)
    motion_max = np.max(frame_diffs)

    # Analyze temporal windows
    window_motion = []
    for i in range(
        0, len(frame_diffs) - window_size + 1, window_size
    ):
        window_var = np.var(frame_diffs[i : i + window_size])
        window_motion.append(window_var)

    return {
        "frame_differences": frame_diffs,
        "motion_variance": motion_variance,
        "motion_mean": motion_mean,
        "motion_max": motion_max,
        "window_motion_variance": window_motion,
        "has_significant_motion": motion_variance
        > 0.1,  # Threshold for "spooky" content
    }


def validate_predictions(
    predictions, ground_truth_labels, categories
):
    """
    Validate model predictions against ground truth categories.
    """
    if len(predictions) != len(ground_truth_labels):
        print(
            "Warning: Prediction and ground truth lengths don't match"
        )
        return {}

    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()

    # Compute accuracy
    correct = (pred_classes == ground_truth_labels).sum()
    accuracy = correct / len(ground_truth_labels)

    # Per-category analysis
    category_stats = {}
    for i, category in enumerate([
        "text",
        "objects",
        "dynamic_scenes",
        "shapes",
    ]):
        mask = ground_truth_labels == i
        if mask.sum() > 0:
            category_acc = (
                pred_classes[mask] == ground_truth_labels[mask]
            ).sum() / mask.sum()
            category_stats[category] = {
                "accuracy": float(category_acc),
                "count": int(mask.sum()),
                "predictions": pred_classes[mask].tolist(),
            }

    return {
        "overall_accuracy": float(accuracy),
        "correct_predictions": int(correct),
        "total_predictions": len(ground_truth_labels),
        "category_stats": category_stats,
        "predicted_classes": pred_classes.tolist(),
        "ground_truth": ground_truth_labels.tolist(),
    }


if __name__ == "__main__":
    main()
