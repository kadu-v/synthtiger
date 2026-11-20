"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component


class DefocusBlur(Component):
    def __init__(self, radius=(1, 2)):
        super().__init__()
        if (
            len(radius) != 2
            or not np.all(np.isfinite(radius))
            or not 0 <= radius[0] <= radius[1]
        ):
            raise ValueError("radius must be a finite nonnegative increasing pair")
        self.radius = radius

    def sample(self, meta=None):
        meta = {} if meta is None else meta
        radius = meta["radius"] if "radius" in meta else np.random.uniform(*self.radius)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        return {"radius": radius}

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        radius = meta["radius"]
        if radius == 0:
            return meta

        extent = int(np.ceil(radius))
        y, x = np.mgrid[-extent : extent + 1, -extent : extent + 1]
        kernel = (x * x + y * y <= radius * radius).astype(np.float32)
        kernel /= kernel.sum()
        for layer in layers:
            layer.image = cv2.filter2D(
                layer.image, -1, kernel, borderType=cv2.BORDER_REFLECT_101
            )
        return meta
