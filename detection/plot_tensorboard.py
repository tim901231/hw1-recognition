#!/usr/bin/env python3
"""Export FCOS TensorBoard curves and images without opening TensorBoard."""
import argparse
import csv
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    accumulator = EventAccumulator(
        str(path), size_guidance={'scalars': 0, 'images': 0},
        purge_orphaned_data=False,
    )
    accumulator.Reload()
    return accumulator


def latest_segment(events, tag):
    # The original trainer can create one event file per step. Read the whole
    # directory, then separate restarted experiments by decreasing step number.
    events = sorted(events, key=lambda event: event.wall_time)
    start = 0
    restarts = 0
    for i in range(1, len(events)):
        if events[i].step < events[i - 1].step:
            start = i
            restarts += 1
    if restarts:
        print(f'{tag}: detected {restarts} restart(s); using the latest segment.')
    return {event.step: event.value for event in events[start:]}


def plot_losses(accumulator, mode, output):
    names = ['cls', 'box', 'ctr']
    tags = [f'train/loss_{name}_{mode}' for name in names]
    available = accumulator.Tags().get('scalars', [])
    if not any(tag in available for tag in tags):
        print(f'No {mode} loss tags found; skipping.')
        return 0
    missing = [tag for tag in tags if tag not in available]
    if missing:
        raise ValueError(f'Missing loss components: {missing}')
    series = [latest_segment(accumulator.Scalars(tag), tag) for tag in tags]
    steps = sorted(set.intersection(*(set(values) for values in series)))
    if not steps:
        raise ValueError(f'No shared steps for {mode} losses.')
    if any(set(values) != set(steps) for values in series):
        print(f'{mode}: plotting only steps shared by all three components.')
    components = np.array([[values[step] for step in steps] for values in series])
    total = components.sum(axis=0)
    if not np.isfinite(components).all():
        print(f'WARNING: {mode} contains NaN/Inf; shown as gaps, retained in CSV.')
    prefix = f'q3_{mode}'
    with (output / f'{prefix}_loss.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'loss_cls', 'loss_box', 'loss_ctr', 'loss_total'])
        writer.writerows(zip(steps, *components, total))
    title = 'Overfit' if mode == 'overfit' else 'Full Training'
    for suffix, curves, labels in [
        ('loss', [total], ['Total loss']),
        ('loss_components', components, ['Classification', 'Box regression', 'Centerness']),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for values, label in zip(curves, labels):
            ax.plot(steps, np.where(np.isfinite(values), values, np.nan), linewidth=2, label=label)
        ax.set(xlabel='Training step', ylabel='Loss', title=f'FCOS {title}: ' + ('Total Loss' if suffix == 'loss' else 'Loss Components'))
        ax.grid(True, alpha=0.3)
        if len(labels) > 1:
            ax.legend()
        fig.tight_layout()
        path = output / f'{prefix}_{suffix}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {path}')
    return 2


def export_images(accumulator, output, all_images=False):
    count = 0
    for tag, filename in [('test_images', 'q3_detections.png'), ('train/gt_images', 'q3_ground_truth.png')]:
        if tag not in accumulator.Tags().get('images', []):
            continue
        events = accumulator.Images(tag)
        if not events:
            continue
        events = sorted(events, key=lambda event: event.wall_time)
        selected = list(enumerate(events)) if all_images else [(len(events) - 1, events[-1])]
        for index, event in selected:
            name = (f'{Path(filename).stem}_{index:03d}_step{event.step}.png'
                    if all_images else filename)
            path = output / name
            with Image.open(BytesIO(event.encoded_image_string)) as image:
                image.convert('RGB').save(path)
            print(f'Saved: {path.resolve()} (tag={tag}, step={event.step})')
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--log_dir', default='detection_logs', help='Event directory or event file.')
    source.add_argument('--event_file', help='Read one specific events.out.tfevents.* file.')
    parser.add_argument('--images_only', action='store_true', help='Export image grids without plotting losses.')
    parser.add_argument('--all_images', action='store_true', help='Export every logged image grid instead of only the latest.')
    parser.add_argument('--mode', choices=['all', 'overfit', 'full'], default='all')
    parser.add_argument('--out_dir', default='../figures')
    args = parser.parse_args()
    accumulator = load_run(args.event_file or args.log_dir)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    modes = ['overfit', 'full'] if args.mode == 'all' else [args.mode]
    count = export_images(accumulator, output, all_images=args.all_images)
    if not count:
        print('No test_images or train/gt_images found in this source. '
              f'Available image tags: {accumulator.Tags().get("images", [])}. '
              'Choose the event file from test inference, or the full detection_logs directory.')
    if not args.images_only:
        count += sum(plot_losses(accumulator, mode, output) for mode in modes)
    if not count:
        parser.error(f'No supported detection tags found. Available tags: {accumulator.Tags()}')
    print(f'Done: {count} PNGs in {output.resolve()}')
    print('For mAP/AP figures, use the separate evaluation output in mAP/output/.')


if __name__ == '__main__':
    main()
