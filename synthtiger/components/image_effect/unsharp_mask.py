"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component


class UnsharpMask(Component):
    def __init__(self, radius=(0.5, 1.5), percent=(100, 200), threshold=(2, 5)):
        super().__init__()
        ranges = (("radius", radius), ("percent", percent), ("threshold", threshold))
        for name, value in ranges:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in value
                )
                or not np.all(np.isfinite(value))
                or not 0 <= value[0] <= value[1]
            ):
                raise ValueError(f"{name} must be a finite nonnegative increasing pair")
            setattr(self, name, value)

    def sample(self, meta=None):
        meta = {} if meta is None else meta
        result = {}
        for name in ("radius", "percent", "threshold"):
            value = meta[name] if name in meta else np.random.uniform(*getattr(self, name))
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
            result[name] = value
        return result

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        if meta["radius"] == 0 or meta["percent"] == 0:
            return meta
        for layer in layers:
            image = layer.image.copy()
            rgb = image[..., :3].astype(np.float32)
            blurred = cv2.GaussianBlur(rgb, (0, 0), meta["radius"])
            detail = rgb - blurred
            image[..., :3] = np.clip(
                rgb + np.where(
                    np.abs(detail) > meta["threshold"],
                    detail * meta["percent"] / 100,
                    0,
                ),
                0,
                255,
            )
            layer.image = image
        return meta
