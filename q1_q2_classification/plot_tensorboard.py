#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# TensorBoard loading
# ============================================================

def load_run(log_dir):
    """Load all scalar and histogram events from TensorBoard."""

    log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(
            f"Log directory does not exist: {log_dir}"
        )

    ea = EventAccumulator(
        str(log_dir),
        size_guidance={
            "scalars": 0,
            "histograms": 0,
        },
    )

    ea.Reload()

    return ea


def get_scalar(ea, tag):
    """Return steps and values for a TensorBoard scalar."""

    scalar_tags = ea.Tags().get("scalars", [])

    if tag not in scalar_tags:
        raise KeyError(
            f"Could not find scalar tag '{tag}'.\n"
            f"Available scalar tags:\n{scalar_tags}"
        )

    events = ea.Scalars(tag)

    steps = [event.step for event in events]
    values = [event.value for event in events]

    return steps, values


# ============================================================
# Scalar plots
# ============================================================

def save_single_plot(
    steps,
    values,
    ylabel,
    title,
    output_path,
):
    plt.figure(figsize=(7, 5))

    plt.plot(
        steps,
        values,
        linewidth=2,
    )

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

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
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Histogram helpers
# ============================================================

def find_histogram_tag(ea, suffix):
    """
    Find a histogram tag by suffix.

    This lets the script work with tags like:

        resnet.layer1.1.conv1.weight/grad

    or

        layer1.1.conv1.weight/grad
    """

    hist_tags = ea.Tags().get("histograms", [])

    # exact match first
    if suffix in hist_tags:
        return suffix

    # otherwise suffix match
    matches = [
        tag
        for tag in hist_tags
        if tag.endswith(suffix)
    ]

    if len(matches) == 1:
        print(
            f"Matched histogram tag:\n"
            f"  requested: {suffix}\n"
            f"  found:     {matches[0]}"
        )

        return matches[0]

    if len(matches) == 0:
        raise KeyError(
            f"Could not find histogram ending with:\n"
            f"  {suffix}\n\n"
            f"Available histogram tags:\n"
            + "\n".join(hist_tags)
        )

    raise KeyError(
        f"Multiple histogram tags match '{suffix}':\n"
        + "\n".join(matches)
    )


def histogram_to_curve(hist):
    """
    Convert TensorBoard histogram buckets into an approximate
    (bin center, normalized count) curve.
    """

    bucket_limits = np.asarray(
        hist.bucket_limit,
        dtype=np.float64,
    )

    counts = np.asarray(
        hist.bucket,
        dtype=np.float64,
    )

    minimum = float(hist.min)
    maximum = float(hist.max)

    centers = []
    valid_counts = []

    previous = minimum

    for upper, count in zip(bucket_limits, counts):

        if not np.isfinite(upper):
            continue

        lower = max(previous, minimum)
        upper_clipped = min(upper, maximum)

        if upper_clipped >= lower and count > 0:
            center = (lower + upper_clipped) / 2.0

            centers.append(center)
            valid_counts.append(count)

        previous = upper

        if upper >= maximum:
            break

    centers = np.asarray(centers)
    valid_counts = np.asarray(valid_counts)

    if valid_counts.sum() > 0:
        valid_counts = valid_counts / valid_counts.sum()

    return centers, valid_counts


def save_histogram_evolution(
    ea,
    tag_suffix,
    title,
    output_path,
    num_snapshots=6,
):
    """
    Plot several gradient histogram snapshots over training.

    This makes it easier to see how the gradient distribution
    changes over time.
    """

    tag = find_histogram_tag(
        ea,
        tag_suffix,
    )

    events = ea.Histograms(tag)

    if len(events) == 0:
        raise RuntimeError(
            f"No histogram events found for '{tag}'"
        )

    # Choose evenly spaced snapshots across training
    n = min(num_snapshots, len(events))

    indices = np.linspace(
        0,
        len(events) - 1,
        n,
        dtype=int,
    )

    # Remove duplicates if there are very few events
    indices = np.unique(indices)

    plt.figure(figsize=(8, 5))

    for idx in indices:

        event = events[idx]

        x, y = histogram_to_curve(
            event.histogram_value
        )

        if len(x) == 0:
            continue

        plt.plot(
            x,
            y,
            linewidth=1.8,
            label=f"Step {event.step}",
        )

    plt.xlabel("Gradient value")
    plt.ylabel("Fraction of gradients")

    plt.title(title)

    plt.legend(
        fontsize=8
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Q1
# ============================================================

def plot_q1(no_aug_dir, aug_dir, best_dir, out_dir):

    print("\n========== Q1 ==========")

    no_aug = load_run(no_aug_dir)
    aug = load_run(aug_dir)
    best = load_run(best_dir)

    # --------------------------------------------------------
    # Figure 1.1
    # Loss with / without augmentation
    # --------------------------------------------------------

    no_aug_loss_steps, no_aug_loss = get_scalar(
        no_aug,
        "Loss/train",
    )

    aug_loss_steps, aug_loss = get_scalar(
        aug,
        "Loss/train",
    )

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

    # --------------------------------------------------------
    # Figure 1.2
    # mAP with / without augmentation
    # --------------------------------------------------------

    no_aug_map_steps, no_aug_map = get_scalar(
        no_aug,
        "map",
    )

    aug_map_steps, aug_map = get_scalar(
        aug,
        "map",
    )

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

    # --------------------------------------------------------
    # Figure 1.3
    # best model loss
    # --------------------------------------------------------

    steps, values = get_scalar(
        best,
        "Loss/train",
    )

    save_single_plot(
        steps,
        values,
        ylabel="Training loss",
        title="Training Loss of Best Simple CNN",
        output_path=out_dir / "q1_best_loss.png",
    )

    # --------------------------------------------------------
    # Figure 1.4
    # best model mAP
    # --------------------------------------------------------

    steps, values = get_scalar(
        best,
        "map",
    )

    save_single_plot(
        steps,
        values,
        ylabel="mAP",
        title="Test mAP of Best Simple CNN",
        output_path=out_dir / "q1_best_map.png",
    )

    # --------------------------------------------------------
    # Figure 1.5
    # learning rate
    # --------------------------------------------------------

    steps, values = get_scalar(
        best,
        "learning_rate",
    )

    save_single_plot(
        steps,
        values,
        ylabel="Learning rate",
        title="Learning Rate of Best Simple CNN",
        output_path=out_dir / "q1_lr.png",
    )


# ============================================================
# Q2
# ============================================================

def plot_q2(q2_best_dir, out_dir):

    print("\n========== Q2 ==========")

    q2 = load_run(q2_best_dir)

    print("\nAvailable scalar tags:")
    print(q2.Tags().get("scalars", []))

    print("\nAvailable histogram tags:")
    for tag in q2.Tags().get("histograms", []):
        print(" ", tag)

    # --------------------------------------------------------
    # Figure 2.1
    # learning rate
    # --------------------------------------------------------

    steps, values = get_scalar(
        q2,
        "learning_rate",
    )

    save_single_plot(
        steps,
        values,
        ylabel="Learning rate",
        title="Learning Rate for ResNet-18",
        output_path=out_dir / "q2_lr.png",
    )

    # --------------------------------------------------------
    # Figure 2.2
    # test mAP
    # --------------------------------------------------------

    steps, values = get_scalar(
        q2,
        "map",
    )

    save_single_plot(
        steps,
        values,
        ylabel="mAP",
        title="Test mAP for ResNet-18",
        output_path=out_dir / "q2_map.png",
    )

    # --------------------------------------------------------
    # Figure 2.3
    # training loss
    # --------------------------------------------------------

    steps, values = get_scalar(
        q2,
        "Loss/train",
    )

    save_single_plot(
        steps,
        values,
        ylabel="Training loss",
        title="Training Loss for ResNet-18",
        output_path=out_dir / "q2_loss.png",
    )

    # --------------------------------------------------------
    # Figure 2.4
    # layer1.1.conv1.weight gradient histogram
    # --------------------------------------------------------

    save_histogram_evolution(
        q2,
        tag_suffix="layer1.1.conv1.weight/grad",
        title="Gradient Distribution: layer1.1.conv1.weight",
        output_path=out_dir / "q2_conv1_grad.png",
    )

    # --------------------------------------------------------
    # Figure 2.5
    # layer4.0.bn2.bias gradient histogram
    # --------------------------------------------------------

    save_histogram_evolution(
        q2,
        tag_suffix="layer4.0.bn2.bias/grad",
        title="Gradient Distribution: layer4.0.bn2.bias",
        output_path=out_dir / "q2_bn4_grad.png",
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate TensorBoard plots for HW1 "
            "Parts 1 and 2."
        )
    )

    # Q1 arguments
    parser.add_argument(
        "--no_aug",
        default=None,
        help="Q1 run WITHOUT augmentation.",
    )

    parser.add_argument(
        "--aug",
        default=None,
        help="Q1 run WITH augmentation.",
    )

    parser.add_argument(
        "--best",
        default=None,
        help="Best Q1 simple CNN run.",
    )

    # Q2 argument
    parser.add_argument(
        "--q2_best",
        default=None,
        help="Best Q2 ResNet-18 TensorBoard run.",
    )

    parser.add_argument(
        "--out_dir",
        default="figures",
        help="Output figure directory.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Q1
    # --------------------------------------------------------

    q1_args = [
        args.no_aug,
        args.aug,
        args.best,
    ]

    if any(x is not None for x in q1_args):

        if not all(x is not None for x in q1_args):
            raise ValueError(
                "For Q1, you must provide all three:\n"
                "  --no_aug\n"
                "  --aug\n"
                "  --best"
            )

        plot_q1(
            args.no_aug,
            args.aug,
            args.best,
            out_dir,
        )

    # --------------------------------------------------------
    # Q2
    # --------------------------------------------------------

    if args.q2_best is not None:

        plot_q2(
            args.q2_best,
            out_dir,
        )

    if (
        args.q2_best is None
        and not any(x is not None for x in q1_args)
    ):
        parser.error(
            "Provide Q1 directories and/or --q2_best."
        )

    print("\nDone.")
    print(
        "Figures saved to:",
        out_dir.resolve(),
    )


if __name__ == "__main__":
    main()