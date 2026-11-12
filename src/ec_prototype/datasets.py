"""Demo for downloading compressed Sentinel-2 datasets."""

import abc
import json
from enum import StrEnum
from pathlib import Path

import georvq
import numpy as np
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from ec_prototype.band_source import open_band_source

REPO_ID = "earth-compress/EC-prototype-S2L1C"

SAMPLE_NAME = "S2A_MSIL1C_20170720T045701_N0500_R119_T44PNB_20230831T180106.SAFE"

DATA_DIR = Path("data")


class DatasetKey(StrEnum):
    """Enum holding keys for each of the dataset versions."""

    GEORVQ = "georvq"
    JPEGXL = "jpegxl"
    SZ3 = "sz3"


class BaseDataset(BaseModel, abc.ABC):
    """The base dataset model."""

    description: str
    path: Path | None = None
    bands: tuple | None = None

    @abc.abstractmethod
    def download(self):
        """Base class download method."""
        pass

    @abc.abstractmethod
    def decode(self, device: str | None = None, quality: str | None = None):
        """Base class decode method."""
        pass


class GeorvqDataset(BaseDataset):
    """GEORVQ dataset class."""

    def download(self):
        """Download method for GEORVQ."""
        self.path = download_dataset(DatasetKey.GEORVQ)

    def decode(self, device: str | None = None, quality: str | None = None):
        """Decode method for GEORVQ."""
        if not self.path:
            raise Exception("No path set, call download() on the dataset first!")
        _bands, _grids = decode_georvq(self.path, device=device, quality=quality)
        self.bands = _bands
        return _grids


class JpegxlDataset(BaseDataset):
    """JPEXGL dataset class."""

    def download(self):
        """Download method for JPEGXL."""
        self.path = download_dataset(DatasetKey.JPEGXL)

    def decode(self, device: str | None = None, quality: str | None = None):
        """Decode method for JPEGXL."""
        if not self.path:
            raise Exception("No path set, call download() on the dataset first!")
        _bands, _grids = decode_tiled(self.path, level=quality)
        self.bands = _bands
        return _grids


def download_dataset(dataset_version: DatasetKey) -> Path:
    """Download a compressed Sentinel-2 SAFE product.

    Args:
        dataset_version (DatasetKey): Enum item identifying the dataset to be run against.

    Returns:
        path (Path): Path of the directory to analyse.
    """
    relative_path = f"{dataset_version.value}/{SAMPLE_NAME}"

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=[f"{relative_path}/**"],
    )

    scene_path = DATA_DIR / relative_path

    if not (scene_path / "COMPRESSED.json").is_file():
        raise FileNotFoundError(f"Compressed SAFE file not found: {scene_path}")

    return scene_path


def decode_georvq(scene_path: Path, device: str | None = None, quality: str | None = None) -> tuple[tuple, dict]:
    """Decode a GEORVQ compressed Sentinel-2 SAFE file.

    Args:
        scene_path (Path): Path to the compressed Sentinel-2 SAFE directory.
        device (str | None, optional): Device used for decoding.
        quality (str | None, optional): The compression quality level to decode.

    Returns:
        tuple[tuple, dict]: The band descriptions and decoded image grids.
    """
    manifest = json.loads((scene_path / "COMPRESSED.json").read_text())
    ds = georvq.open(scene_path / manifest["store"], device=device, bf16=bool(manifest.get("bf16", False)))
    return ds.descriptions, ds.read_grids(quality=quality)


def decode_tiled(scene_path: Path, level: str | None = None) -> tuple[tuple, dict]:
    """Decode a JPEG XL or SZ3 compressed Sentinel-2 SAFE file.

    Args:
        scene_path (Path): Path to the compressed Sentinel-2 SAFE directory.
        level (str | None, optional): The compression level to decode.

    Returns:
        tuple[tuple, dict]: The band names and decoded image grids.
    """
    source = open_band_source(scene_path, level=level)

    if source.name not in ("jpegxl", "sz3"):
        raise ValueError(f"Expected jpegxl or sz3, got {source.name}")

    bands = tuple(source.bands)
    decoded = source.read(list(bands))

    grids = {}
    for resolution in (10, 20, 60):
        resolution_bands = [band for band in bands if source.native_resolution(band) == resolution]

        if not resolution_bands:
            continue

        # Format the array for plotting outputs in the Jupyter notebook.
        grids[f"{resolution}m"] = np.stack([decoded[band] for band in resolution_bands], axis=0)

    return bands, grids


DATASETS: dict[DatasetKey, BaseDataset] = {
    DatasetKey.GEORVQ: GeorvqDataset(
        description="GeoRVQ compressed Sentinel-2 data",
    ),
    DatasetKey.JPEGXL: JpegxlDataset(description="JPEG XL compressed Sentinel-2 data"),
}
