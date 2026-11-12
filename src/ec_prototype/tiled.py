"""Tiled SZ3 / JPEG XL band data as a band source.

This code is using: https://github.com/asterisk-labs/clouds-decoded/blob/compressed-readers/src/shared_utils/data/compressed/tiled.py


A product holds one or more levels of one codec, each in
``IMG_DATA/<level>/`` with its own codec parameters, listed in the
manifest's ``levels``. Each band's native grid is cut into square tiles (``tile`` B02 pixels on
a side, the same ground area at every resolution) and every tile is coded
on its own, so a crop window decodes only the tiles it touches. Bands of
one resolution group share a file, ``IMG_DATA/<level>/<group>.bin``, a plain
concatenation of coded blobs located by the manifest's index.

Layouts:

- ``separate``: each band's tile is one 2-D blob (SZ3 or JPEG XL).
- ``grouped``: the tile of every band of a resolution group is one 3-D
  ``(bands, rows, cols)`` SZ3 blob, so SZ3 also predicts across bands.
  Asking for one band of a group decodes all of them.

Groups are the bands sharing a native grid (10 m, 20 m, 60 m; on newer
processing baselines B01 lands with the 20 m bands).

Nodata (DN 0) does not survive lossy coding, so each tile of a band
holding any also stores its zero mask, packed bits deflated, and the
reader puts the zeros back. Before coding, a tile's zeros are filled with
the mean of its valid pixels so the edge costs no bits, and decoded valid
pixels are clipped to at least 1 DN.

Codec parameters, as in ``imagecodecs``:

- ``sz3``: ``abs``, the absolute error bound in DN (bands are coded as
  float32 and rounded back to uint16).
- ``jpegxl``: ``distance`` (Butteraugli distance; 0 codes losslessly) and
  ``effort`` (1-10, default 7).
"""
from __future__ import annotations

import os
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from ec_prototype.band_source import BandGeoref, BandSource, require_optional

CODECS = ("sz3", "jpegxl")
LAYOUTS = ("separate", "grouped")
DEFAULT_TILE = 1098  # B02 pixels; 10 x 10 tiles per product, 549 px at 20 m, 183 px at 60 m
_JOINT = "*"         # index key of a grouped blob


def _imagecodecs(codec: str):
    ic = require_optional("imagecodecs", f"read and write {codec}-compressed Sentinel-2 bands",
                          "pip install imagecodecs")
    available = {"sz3": ic.SZ3.available, "jpegxl": ic.JPEGXL.available}[codec]
    if not available:
        raise ImportError(f"this imagecodecs build ({ic.__version__}) has no {codec} codec")
    return ic


def _decode(ic, codec: str, blob: bytes, shape: tuple[int, ...]) -> np.ndarray:
    if codec == "sz3":
        a = ic.sz3_decode(blob, shape=shape, dtype=np.float32)
        return np.clip(np.rint(a), 0, 65535).astype(np.uint16)
    return np.asarray(ic.jpegxl_decode(blob), np.uint16).reshape(shape)


def _unpack(blob: bytes, shape: tuple[int, int]) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(zlib.decompress(blob), np.uint8))
    return bits[: shape[0] * shape[1]].reshape(shape).astype(bool)


class _Grid:
    """The tiling of one group's native grid."""

    def __init__(self, shape: tuple[int, int], tile_px: int):
        self.shape, self.tile_px = tuple(shape), int(tile_px)
        self.ny = -(-self.shape[0] // self.tile_px)
        self.nx = -(-self.shape[1] // self.tile_px)

    def bounds(self, ty: int, tx: int) -> tuple[int, int, int, int]:
        t = self.tile_px
        return (ty * t, min((ty + 1) * t, self.shape[0]), tx * t, min((tx + 1) * t, self.shape[1]))

    def tiles(self, r0: int, r1: int, c0: int, c1: int) -> Iterable[tuple[int, int]]:
        t = self.tile_px
        for ty in range(r0 // t, (r1 - 1) // t + 1):
            for tx in range(c0 // t, (c1 - 1) // t + 1):
                yield ty, tx


class TiledBandSource(BandSource):
    """Tiled SZ3 or JPEG XL band data written by ``write_tiled_safe``."""

    def __init__(self, scene_directory: Path, manifest: dict[str, Any]):
        self.codec = manifest["codec"]["name"]
        if self.codec not in CODECS:
            raise ValueError(f"unknown tiled codec '{self.codec}'; known: {CODECS}")
        self.name = self.codec
        self.params = manifest["codec"].get("params", {})
        self.layout = manifest["layout"]
        self._ic = _imagecodecs(self.codec)
        self.root = Path(scene_directory)
        g = manifest["georef"]
        self._georef = BandGeoref(CRS.from_wkt(g["crs"]), Affine(*g["transform"]),
                                  int(g["width"]), int(g["height"]))
        self._bands = manifest["bands"]
        self._groups = manifest["groups"]
        self._grids = {name: _Grid(grp["shape"], grp["tile_px"]) for name, grp in self._groups.items()}
        self.manifest = manifest
        self.n_workers = min(32, os.cpu_count() or 1)

    @property
    def groups(self) -> dict[str, list[str]]:
        if self.layout == "grouped":
            return {name: list(grp["bands"]) for name, grp in self._groups.items()}
        return {b: [b] for grp in self._groups.values() for b in grp["bands"]}

    def georef(self) -> BandGeoref:
        return self._georef

    def shape(self, band: str) -> tuple[int, int]:
        return tuple(self._bands[band]["shape"])

    def native_resolution(self, band: str) -> int:
        return int(self._bands[band]["res"])

    def _group_of(self, band: str) -> str:
        return next(n for n, grp in self._groups.items() if band in grp["bands"])

    def _read_group(self, group: str, windows: dict[str, Window | None]) -> dict[str, np.ndarray]:
        bands = self.groups[group]
        gname = self._group_of(bands[0])
        grp, grid = self._groups[gname], self._grids[gname]
        w = windows[bands[0]]
        if w is None:
            r0, r1, c0, c1 = 0, grid.shape[0], 0, grid.shape[1]
        else:
            r0, c0 = int(w.row_off), int(w.col_off)
            r1, c1 = r0 + int(w.height), c0 + int(w.width)
        out = {b: np.zeros((r1 - r0, c1 - c0), np.uint16) for b in bands}
        with open(self.root / grp["file"], "rb") as f:
            def fetch(entry):
                f.seek(entry[0])
                return f.read(entry[1])
            jobs = []
            for ty, tx in grid.tiles(r0, r1, c0, c1):
                idx = grp["index"].get(f"{ty}_{tx}", {})
                blobs = {k: fetch(v) for k, v in idx.get("data", {}).items()}
                masks = {k: fetch(v) for k, v in idx.get("mask", {}).items()}
                jobs.append((grid.bounds(ty, tx), blobs, masks))

        def decode(job):
            (y0, y1, x0, x1), blobs, masks = job
            shape = (y1 - y0, x1 - x0)
            if _JOINT in blobs:
                stack = _decode(self._ic, self.codec, blobs[_JOINT], (len(grp["bands"]),) + shape)
                tiles = dict(zip(grp["bands"], stack))
            else:
                tiles = {b: _decode(self._ic, self.codec, blobs[b], shape) for b in bands if b in blobs}
            res = {}
            for b in bands:
                t = tiles.get(b)
                if t is None:  # all nodata
                    res[b] = np.zeros(shape, np.uint16)
                    continue
                t = np.maximum(t, 1)
                if b in masks:
                    t[_unpack(masks[b], shape)] = 0
                res[b] = t
            return (y0, y1, x0, x1), res

        with ThreadPoolExecutor(max(1, min(self.n_workers, len(jobs)))) as ex:
            for (y0, y1, x0, x1), res in ex.map(decode, jobs):
                ys, ye = max(y0, r0), min(y1, r1)
                xs, xe = max(x0, c0), min(x1, c1)
                for b, t in res.items():
                    out[b][ys - r0:ye - r0, xs - c0:xe - c0] = t[ys - y0:ye - y0, xs - x0:xe - x0]
        return out

    def describe(self) -> dict[str, Any]:
        return {"reader": self.name, "codec": self.codec, "level": self.manifest.get("level"),
                "layout": self.layout,
                "params": self.params, "bits_per_pixel": self.manifest.get("bits_per_pixel")}
