"""Where a scene's band images come from.

Adapted from Clouds Decoded: https://github.com/asterisk-labs/clouds-decoded
https://github.com/asterisk-labs/clouds-decoded/blob/compressed-readers/src/shared_utils/data/band_source.py

The decode functionality has been used in this demo.

"""

from __future__ import annotations

import importlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from rasterio.windows import Window

logger = logging.getLogger(__name__)

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

MANIFEST_NAME = "COMPRESSED.json"
MANIFEST_FORMAT = "clouds-decoded-compressed"
MANIFEST_VERSION = 2  # 2 adds levels; version 1 manifests (one level) still read

# reader name in a manifest -> "module:class", imported on first use
READERS: dict[str, str] = {
    "sz3": "ec_prototype.compressed.tiled:TiledBandSource",
    "jpegxl": "ec_prototype.compressed.tiled:TiledBandSource",
}


@dataclass(frozen=True)
class BandGeoref:
    """Georeferencing of the B02 (10 m) reference grid of a full product."""

    crs: Any
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` in CRS units."""
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.width, self.height)
        return left, bottom, right, top


def scale_crop_window(
    crop_window: tuple,
    b02_w: int,
    b02_h: int,
    src_w: int,
    src_h: int,
) -> Window:
    """Scale a B02-pixel crop window to the coordinate space of a band raster.

    Args:
        crop_window: ``(col_off, row_off, width, height)`` in B02 (10 m) pixels.
        b02_w: Full-scene B02 pixel width (from the actual file, not BAND_RESOLUTIONS).
        b02_h: Full-scene B02 pixel height.
        src_w: Pixel width of the target band raster.
        src_h: Pixel height of the target band raster.

    Returns:
        A rasterio ``Window`` in the target band's pixel coordinates.
    """
    col_off, row_off, width, height = crop_window
    scale_x = src_w / b02_w
    scale_y = src_h / b02_h
    return Window(
        col_off=int(col_off * scale_x),
        row_off=int(row_off * scale_y),
        width=int(width * scale_x),
        height=int(height * scale_y),
    )


def require_optional(module: str, purpose: str, install: str):
    """Import an optional dependency, or explain what is missing.

    Args:
        module: Module to import, e.g. ``"imagecodecs"``.
        purpose: What it is needed for, completing "... is needed to <purpose>".
        install: How to install it.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"The '{module}' package is needed to {purpose}, but it could not be "
            f"imported ({e}). It is not a clouds-decoded dependency; install it "
            f"with: {install}"
        ) from e


class BandSource(ABC):
    """The band images of one product, decoded a group at a time.

    Subclasses define ``groups`` (decode units, group name -> band names),
    ``georef()``, ``shape()`` and ``_read_group()``.
    """

    #: Short name of the representation, e.g. ``"jp2"`` or ``"sz3"``.
    name: str = "unknown"

    @property
    @abstractmethod
    def groups(self) -> dict[str, list[str]]:
        """Decode units: group name -> the bands decoded together."""

    @abstractmethod
    def georef(self) -> BandGeoref:
        """CRS, transform and size of the full product's B02 grid."""

    @abstractmethod
    def shape(self, band: str) -> tuple[int, int]:
        """``(height, width)`` of a band's full native grid."""

    @abstractmethod
    def _read_group(self, group: str, windows: dict[str, Window | None]) -> dict[str, np.ndarray]:
        """Decode one group.

        Args:
            group: A key of ``groups``.
            windows: Every band of the group -> its native-grid window, or
                ``None`` for the full band. All bands of a group cover the
                same area.

        Returns:
            Band name -> uint16 DN array, for every band of the group.
        """

    @property
    def bands(self) -> list[str]:
        """All bands the source holds, in canonical order."""
        held = {b for members in self.groups.values() for b in members}
        return [b for b in BANDS if b in held]

    def native_resolution(self, band: str) -> int:
        """Actual pixel size of a band in metres, from its grid size."""
        ref = self.georef()
        return round(abs(ref.transform.a) * ref.width / self.shape(band)[1])

    def window(self, band: str, crop_window: tuple | None) -> Window | None:
        """A B02-pixel crop window on a band's native grid (``None`` for none)."""
        if not crop_window:
            return None
        ref = self.georef()
        h, w = self.shape(band)
        return scale_crop_window(crop_window, ref.width, ref.height, w, h)

    def read(self, bands: list[str], crop_window: tuple | None = None) -> dict[str, np.ndarray]:
        """Read bands, decoding each group that holds one of them once.

        Args:
            bands: Band names to return.
            crop_window: Optional ``(col_off, row_off, width, height)`` in B02 pixels.

        Returns:
            Band name -> DN array on the band's native grid, for ``bands``.

        Raises:
            KeyError: If a band is not held by this source.
        """
        group_of = {b: g for g, members in self.groups.items() for b in members}
        missing = [b for b in bands if b not in group_of]
        if missing:
            raise KeyError(f"{self.name} source has no band(s) {missing}; it holds {self.bands}")
        out: dict[str, np.ndarray] = {}
        for group in dict.fromkeys(group_of[b] for b in bands):
            windows = {b: self.window(b, crop_window) for b in self.groups[group]}
            decoded = self._read_group(group, windows)
            out.update({b: a for b, a in decoded.items() if b in bands})
        return {b: out[b] for b in bands}

    def describe(self) -> dict[str, Any]:
        """Provenance of the representation, e.g. codec and level."""
        return {"reader": self.name}


def read_manifest(scene_directory: Path) -> dict[str, Any] | None:
    """The product's ``COMPRESSED.json``, or ``None`` for a standard product.

    Raises:
        ValueError: If the manifest is not a clouds-decoded compressed manifest
            or is of a newer version than this code reads.
    """
    path = Path(scene_directory) / MANIFEST_NAME
    if not path.exists():
        return None
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path} is not a {MANIFEST_FORMAT} manifest")
    if int(manifest.get("version", 0)) > MANIFEST_VERSION:
        raise ValueError(
            f"{path} is manifest version {manifest['version']}; this clouds-decoded "
            f"reads up to version {MANIFEST_VERSION}"
        )
    return manifest


def levels(manifest: dict[str, Any]) -> list[str]:
    """The level names of a manifest (empty for a single-level one)."""
    return list(manifest.get("levels", {}))


def resolve_level(manifest: dict[str, Any], level: str | None = None) -> dict[str, Any]:
    """The manifest as seen by a reader for one level: the product-wide
    fields, overlaid with the level's own, plus ``level``.

    Args:
        manifest: A product manifest.
        level: Level to read; ``None`` for the manifest's default.

    Returns:
        The resolved manifest.

    Raises:
        ValueError: If the level is not in the manifest.
    """  # ruff: ignore[missing-blank-line-after-summary]
    if "levels" not in manifest:
        if level is not None and level != manifest.get("level"):
            raise ValueError(f"product has a single level ({manifest.get('level')}); cannot read level '{level}'")
        return manifest
    name = manifest.get("default") if level is None else level
    if name not in manifest["levels"]:
        raise ValueError(f"no level '{name}' in the product; it has {levels(manifest)}")
    out = {k: v for k, v in manifest.items() if k not in ("levels", "default")}
    out.update(manifest["levels"][name])
    out["level"] = name
    return out


def open_band_source(scene_directory: Path, band_paths=None, level: str | None = None) -> BandSource:
    """The band source of a product directory.

    Args:
        scene_directory: The ``.SAFE`` directory.
        band_paths: Callable ``(bands) -> {band: path}`` locating the JPEG 2000
            images of a standard product.
        level: For a compressed product with several levels, the level to
            read (``None``: the manifest's default).

    Returns:
        A ``Jp2BandSource`` for a standard product, otherwise the reader the
        product's manifest names.

    Raises:
        ValueError: If the manifest names an unknown reader or level, or a
            level is asked of a standard product.
        ImportError: If the reader needs a package that is not installed.
    """
    scene_directory = Path(scene_directory)
    manifest = read_manifest(scene_directory)
    if manifest is None:
        raise FileNotFoundError(f"Compressed SAFE manifest not found: {scene_directory / MANIFEST_NAME}")
    manifest = resolve_level(manifest, level)
    reader = manifest.get("reader")
    if reader not in READERS:
        raise ValueError(
            f"{scene_directory / MANIFEST_NAME} names reader '{reader}'; known readers are {sorted(READERS)}"
        )
    module, cls = READERS[reader].split(":")
    source = getattr(importlib.import_module(module), cls)(scene_directory, manifest)
    logger.info("Reading %s band images through the %s reader (%s)", scene_directory.name, reader, source.describe())
    return source
