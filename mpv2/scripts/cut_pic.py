#!/usr/bin/env python3
"""Crop a batch of screenshots using the original hard-coded bounds."""

from __future__ import annotations

from pathlib import Path

import cv2


SOURCE_DIR = Path("/home/msi/Pictures")
OUTPUT_DIR = Path("../output")
ROW_SLICE = slice(350, 1620)
COLUMN_SLICE = slice(150, 2272 - 150)


def crop_series(start: int = 1, stop: int = 20) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(start, stop + 1):
        image_path = SOURCE_DIR / f"s{index}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        print(image.shape)
        cropped = image[ROW_SLICE, COLUMN_SLICE]
        cv2.imwrite(str(OUTPUT_DIR / f"s{index}.png"), cropped)


if __name__ == "__main__":
    crop_series()
