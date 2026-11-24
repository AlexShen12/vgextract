"""Accelerate the existing static SKN projection when Numba is installed."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from PIL import Image

from .body_static_skn import render_view as reference_render


def rasterize(projected, uv, faces, texture, pixels, depth):
    height, width = texture.shape[:2]
    for first, second, third in faces:
        ax, ay, az = projected[first]
        bx, by, bz = projected[second]
        cx, cy, cz = projected[third]
        abx, aby, abz = bx - ax, by - ay, bz - az
        acx, acy, acz = cx - ax, cy - ay, cz - az
        det = abx * acy - aby * acx
        if abs(det) < 1e-8:
            continue
        left = math.floor(min(ax, bx, cx))
        right = math.ceil(max(ax, bx, cx))
        top = math.floor(min(ay, by, cy))
        bottom = math.ceil(max(ay, by, cy))
        u, v = uv[first]
        bu, bv = uv[second] - uv[first]
        cu, cv = uv[third] - uv[first]
        for y in range(top, bottom):
            dy = y + .5 - ay
            for x in range(left, right):
                dx = x + .5 - ax
                wb = (dx * acy - dy * acx) / det
                wc = (abx * dy - aby * dx) / det
                if wb < -1e-8 or wc < -1e-8 or wb + wc > 1 + 1e-8:
                    continue
                z = az + wb * abz + wc * acz
                if z <= depth[y, x]:
                    continue
                tx = min(width - 1, max(0, int((u + wb * bu + wc * cu) * width)))
                ty = min(height - 1, max(0, int((v + wb * bv + wc * cv) * height)))
                if texture[ty, tx, 3] > 127:
                    depth[y, x] = z
                    pixels[y, x] = texture[ty, tx]


@lru_cache(maxsize=1)
def implementation():
    try:
        import numba
    except ImportError:
        return None, "numpy"
    return numba.njit(rasterize), f"numba:{numba.__version__}"


def accelerated_render(xyz, uv, groups, textures, heading, elevation, size, raster):
    theta, phi = map(math.radians, (heading, elevation))
    view = np.array([math.sin(theta) * math.cos(phi), math.sin(phi), math.cos(theta) * math.cos(phi)])
    right = np.array([math.cos(theta), 0, -math.sin(theta)])
    projected = xyz @ np.stack([right, -np.cross(view, right), view], axis=1)
    used = np.concatenate([faces.ravel() for faces in groups.values()])
    lower, upper = projected[used, :2].min(axis=0), projected[used, :2].max(axis=0)
    span = max(upper - lower)
    if span <= 0:
        raise ValueError("SKN geometry projects to an empty view")
    projected[:, :2] = (projected[:, :2] - (upper + lower) / 2) * (.82 * size / span) + size / 2
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    depth = np.full((size, size), -np.inf)
    for name, faces in groups.items():
        raster(projected, uv, faces, textures[name], pixels, depth)
    return Image.fromarray(pixels)


def render_view(xyz, uv, groups, textures, heading, elevation, size=224):
    raster, _ = implementation()
    if raster is None:
        return reference_render(xyz, uv, groups, textures, heading, elevation, size)
    return accelerated_render(xyz, uv, groups, textures, heading, elevation, size, raster)
