"""Demo for downloading compressed Sentinel-2 datasets."""

import abc
import json
from enum import StrEnum
from pathlib import Path

import georvq
from huggingface_hub import snapshot_download
from pydantic import BaseModel

REPO_ID = "earth-compress/EC-prototype-S2L1C"

SAMPLE_NAME = "S2A_MSIL1C_20170720T045701_N0500_R119_T44PNB_20230831T180106.SAFE"

DATA_DIR = Path("data")


class DatasetKey(StrEnum):
    """Enum holding keys for each of the dataset versions."""

    GEORVQ = "georvq"
    JPEGXL = "jpegxl"
    SZ3 = "sz3"


class BaseDataset(BaseModel, abc.ABC):
    """Abstract base class for Datasets."""

    description: str
    path: Path | None = None
    bands: tuple | None = None

    @abc.abstractmethod
    def download(self):
        """Method to download the dataset."""
        pass

    @abc.abstractmethod
    def decode(self, device: str | None = None):
        """Method to decode the dataset."""
        pass


class GeorvqDataset(BaseDataset):
    """Class for GEORVQ Compressed datasets."""

    def download(self):
        """Download the GEORVQ dataset."""
        self.path = download_dataset(DatasetKey.GEORVQ)

    def decode(self, device: str | None = None, quality: str | None = None):
        """Decode the dataset.

        Args:
            device (str | None, optional): Compute device, ('cuda', 'cpu', 'mps', or None=auto).
            quality (str | None, optional): Quality level to decompress to, e.g. "8-4".
                                            Will use codec's default if not provided.
        """
        if not self.path:
            raise Exception("No path set, call download() on the dataset first!")
        manifest = json.loads((self.path / "COMPRESSED.json").read_text())
        ds = georvq.open(self.path / manifest["store"], device=device, bf16=bool(manifest.get("bf16", False)))
        self.bands = ds.descriptions
        return ds.read_grids(quality=quality)


# class JpegxlDataset(BaseDataset):
#     def download(self):
#         self.path = download_dataset(DatasetKey.JPEGXL)

#     def decode(self):
#         pass


# class Sz3Dataset(BaseDataset):
#     def download(self):
#         self.path = download_dataset(DatasetKey.SZ3)

#     def decode(self):
#         pass


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


DATASETS: dict[DatasetKey, BaseDataset] = {
    DatasetKey.GEORVQ: GeorvqDataset(
        description="GeoRVQ compressed Sentinel-2 data",
    ),
    # DatasetKey.JPEGXL: JpegxlDataset(
    #     description="JPEG XL compressed Sentinel-2 data"
    # ),
    # DatasetKey.SZ3: Sz3Dataset(
    #     description="SZ3 compressed Sentinel-2 data",
    # )
}
