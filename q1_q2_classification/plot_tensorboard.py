#!/usr/bin/env python3
"""
Plot all figures required for Part 1 (PASCAL multi-label classification)
from TensorBoard event files.

Required outputs:
  1. q1_loss_aug_comparison.png
  2. q1_map_aug_comparison.png
  3. q1_best_loss.png
  4. q1_best_map.png
  5. q1_lr.png

Example:
    python plot_part1.py \
        --no_aug runs/q1_no_aug \
        --aug runs/q1_aug \
        --best runs/q1_aug \
        --out_dir figures

If your best run is a different directory, pass that directory to --best.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run(log_dir):
    """Load all scalar events from a TensorBoard run directory."""
    log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    ea = EventAccumulator(
        str(log_dir),
        size_guidance={
            "scalars": 0,  # load all scalar points
        },
    )
    ea.Reload()
    return ea


def get_scalar(ea, tag):
    """Return TensorBoard scalar steps and values for one tag."""
    scalar_tags = ea.Tags().get("scalars", [])

    if tag not in scalar_tags:
        raise KeyError(
            f"Could not find scalar tag '{tag}'.\n"
            f"Available scalar tags: {scalar_tags}"
        )

    events = ea.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values


def save_single_plot(steps, values, ylabel, title, output_path):
    """Save one scalar curve."""
    plt.figure(figsize=(7, 5))
    plt.plot(steps, values, linewidth=2)

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def save_comparison_plot(
    steps_a,
    values_a,
    label_a,
    steps_b,
    values_b,
    label_b,
    ylabel,
    title,
    output_path,
):
    """Save two runs in the same figure."""
    plt.figure(figsize=(7, 5))

    plt.plot(
        steps_a,
        values_a,
        linewidth=2,
        label=label_a,
    )
    plt.plot(
        steps_b,
        values_b,
        linewidth=2,
        label=label_b,
    )

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot every TensorBoard figure required for HW1 Part 1."
    )
    parser.add_argument(
        "--no_aug",
        required=True,
        help="TensorBoard log directory for the run WITHOUT data augmentation.",
    )
    parser.add_argument(
        "--aug",
        required=True,
        help="TensorBoard log directory for the run WITH data augmentation.",
    )
    parser.add_argument(
        "--best",
        required=True,
        help="TensorBoard log directory for your best Part 1 model.",
    )
    parser.add_argument(
        "--out_dir",
        default="figures",
        help="Directory in which to save the plots. Default: figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TensorBoard runs...")
    no_aug = load_run(args.no_aug)
    aug = load_run(args.aug)
    best = load_run(args.best)

    # ------------------------------------------------------------
    # Figure 1.1: Loss/Train with and without augmentation
    # ------------------------------------------------------------
    no_aug_loss_steps, no_aug_loss = get_scalar(no_aug, "Loss/train")
    aug_loss_steps, aug_loss = get_scalar(aug, "Loss/train")

    save_comparison_plot(
        no_aug_loss_steps,
        no_aug_loss,
        "Without augmentation",
        aug_loss_steps,
        aug_loss,
        "With augmentation",
        ylabel="Training loss",
        title="Training Loss: With vs. Without Data Augmentation",
        output_path=out_dir / "q1_loss_aug_comparison.png",
    )

    # ------------------------------------------------------------
    # Figure 1.2: mAP with and without augmentation
    # ------------------------------------------------------------
    no_aug_map_steps, no_aug_map = get_scalar(no_aug, "map")
    aug_map_steps, aug_map = get_scalar(aug, "map")

    save_comparison_plot(
        no_aug_map_steps,
        no_aug_map,
        "Without augmentation",
        aug_map_steps,
        aug_map,
        "With augmentation",
        ylabel="mAP",
        title="Test mAP: With vs. Without Data Augmentation",
        output_path=out_dir / "q1_map_aug_comparison.png",
    )

    # ------------------------------------------------------------
    # Figure 1.3: Training loss for best model
    # ------------------------------------------------------------
    best_loss_steps, best_loss = get_scalar(best, "Loss/train")

    save_single_plot(
        best_loss_steps,
        best_loss,
        ylabel="Training loss",
        title="Training Loss of Best Simple CNN",
        output_path=out_dir / "q1_best_loss.png",
    )

    # ------------------------------------------------------------
    # Figure 1.4: mAP for best model
    # ------------------------------------------------------------
    best_map_steps, best_map = get_scalar(best, "map")

    save_single_plot(
        best_map_steps,
        best_map,
        ylabel="mAP",
        title="Test mAP of Best Simple CNN",
        output_path=out_dir / "q1_best_map.png",
    )

    # ------------------------------------------------------------
    # Figure 1.5: Learning rate for best model
    # ------------------------------------------------------------
    best_lr_steps, best_lr = get_scalar(best, "learning_rate")

    save_single_plot(
        best_lr_steps,
        best_lr,
        ylabel="Learning rate",
        title="Learning Rate of Best Simple CNN",
        output_path=out_dir / "q1_lr.png",
    )

    print("\nDone. Generated all 5 Part 1 figures in:")
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
