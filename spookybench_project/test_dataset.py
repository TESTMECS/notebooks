#!/usr/bin/env python3
"""
Simple test script to explore SpookyBench dataset structure
and understand the data format.
"""

from datasets import load_dataset
from time_decoder import extract_frames_from_video_bytes


def explore_spookybench():
    """Explore the structure of SpookyBench dataset."""

    print("Loading SpookyBench dataset...")
    try:
        # Load just a small sample first
        dataset = load_dataset(
            "timeblindness/spooky-bench",
            name="default",
            split="train[:5]",
        )
        print(f"Successfully loaded {len(dataset)} samples")

        # Examine the first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nFirst sample structure:")
            print(f"Keys: {list(sample.keys())}")

            for key, value in sample.items():
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")

                if hasattr(value, "__len__") and not isinstance(
                    value, (str, bytes)
                ):
                    try:
                        print(f"  Length: {len(value)}")
                    except:
                        pass

                if isinstance(value, dict):
                    print(f"  Dict keys: {list(value.keys())}")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {type(subvalue)}")
                        if isinstance(subvalue, bytes):
                            print(
                                f"      Bytes length: {len(subvalue)}"
                            )
                        elif hasattr(
                            subvalue, "__len__"
                        ) and not isinstance(subvalue, (str, bytes)):
                            try:
                                print(
                                    f"      Length: {len(subvalue)}"
                                )
                            except:
                                pass
                elif isinstance(value, bytes):
                    print(f"  Bytes length: {len(value)}")
                elif isinstance(value, str):
                    print(
                        f"  String content (first 100 chars): {value[:100]}..."
                    )
                else:
                    print(f"  Value: {value}")

        # Show categories and sample distribution
        print("\nDataset info:")
        print(f"Total samples: {len(dataset)}")

        # Try to identify categories if available
        if len(dataset) > 0 and "category" in dataset[0]:
            categories = {}
            for sample in dataset:
                cat = sample.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            print(f"Categories found: {categories}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "Make sure you have internet connection and datasets library installed"
        )
        return False

    return True


def test_video_processing():
    """Test video processing on a small sample."""

    try:

        print("\n" + "=" * 50)
        print("Testing video processing...")

        # Load one sample
        dataset = load_dataset(
            "timeblindness/spooky-bench",
            name="default",
            split="train[:1]",
        )

        if len(dataset) > 0:
            sample = dataset[0]

            if "file" in sample and "bytes" in sample["file"]:
                video_bytes = sample["file"]["bytes"]
                print(
                    f"Processing video of size: {len(video_bytes)} bytes"
                )

                frames = extract_frames_from_video_bytes(video_bytes)

                if frames:
                    print(
                        f"Successfully extracted {len(frames)} frames"
                    )
                    print(f"Frame shape: {frames[0].shape}")
                else:
                    print("No frames extracted")
            else:
                print("No video bytes found in sample")

    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error in video processing test: {e}")


if __name__ == "__main__":
    print("SpookyBench Dataset Explorer")
    print("=" * 50)

    # Explore dataset structure
    success = explore_spookybench()

    if success:
        # Test video processing
        test_video_processing()

    print("\nDone!")
