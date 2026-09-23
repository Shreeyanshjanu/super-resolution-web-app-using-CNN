"""Export a shared-contrast visual comparison without sharpening the images."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from frontend.components.results import read_rgb
from inference.preview import rgb_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("detailed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original, limits = rgb_image(read_rgb(args.crop.read_bytes()))
    previous, _ = rgb_image(read_rgb(args.previous.read_bytes()), limits)
    detailed, _ = rgb_image(read_rgb(args.detailed.read_bytes()), limits)
    original = original.resize(detailed.size, Image.Resampling.BICUBIC)
    if previous.size != detailed.size:
        raise ValueError("Both results must cover the same crop at the same dimensions.")
    width, height = detailed.size
    canvas = Image.new("RGB", (width * 3, height + 50), "#111827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf", 20)
    for index, (label, image) in enumerate(zip(
            ("Original (bicubic enlarged)", "Previous 20-step result", "New 100-step result"),
            (original, previous, detailed))):
        canvas.paste(image, (index * width, 50), image.getchannel("A"))
        draw.text((index * width + 12, 12), label, fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
