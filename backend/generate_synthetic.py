import argparse
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


def make_clip(out_dir: Path, frames: int, image_size: int, color: Tuple[int, int, int], velocity: Tuple[int, int], radius: int = 18) -> None:
    x = random.randint(radius, image_size - radius)
    y = random.randint(radius, image_size - radius)
    vx, vy = velocity
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        img = Image.new("RGB", (image_size, image_size), (15, 20, 35))
        draw = ImageDraw.Draw(img)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        img.save(out_dir / f"frame_{i:04d}.jpg", quality=95)
        x += vx
        y += vy
        if x < radius or x > image_size - radius:
            vx = -vx
            x += 2 * vx
        if y < radius or y > image_size - radius:
            vy = -vy
            y += 2 * vy


def generate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    colors = [(255, 96, 64), (64, 196, 255), (144, 255, 128), (255, 220, 64)]
    velocities = [(3, 0), (0, 3), (2, 2), (-3, 2), (2, -3)]

    for cls_idx, cls in enumerate(args.classes):
        cls_dir = Path(args.output) / cls
        for clip_idx in range(args.clips_per_class):
            clip_dir = cls_dir / f"clip_{clip_idx:03d}"
            color = colors[cls_idx % len(colors)]
            velocity = random.choice(velocities)
            make_clip(
                clip_dir,
                frames=args.frames_per_clip,
                image_size=args.image_size,
                color=color,
                velocity=velocity,
                radius=args.radius,
            )
    print(f"Synthetic dataset written to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic action dataset (moving blobs)")
    parser.add_argument("output", type=str, help="Output directory, e.g., ../data_synth")
    parser.add_argument("--classes", type=str, nargs="+", default=["class_a", "class_b", "class_c"], help="List of class names")
    parser.add_argument("--clips-per-class", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--radius", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
