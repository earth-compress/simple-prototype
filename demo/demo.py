"""Demo for downloading compressed Sentinel-2 datasets."""

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from time import time

import humanize
from huggingface_hub import snapshot_download
from pydantic import BaseModel

REPO_ID = "earth-compress/EC-prototype-S2L1C"

SAMPLE_NAME = "S2A_MSIL1C_20170720T045701_N0500_R119_T44PNB_20230831T180106.SAFE"

DATA_DIR = Path("data")


class DatasetVersion(BaseModel):
    """Model to hold information on each compression-level version of the datasets."""

    download_method: Callable[[], Path]
    description: str


class DatasetKey(StrEnum):
    """Enum holding keys for each of the dataset versions."""

    GEORVQ = "georvq"
    JPEGXL = "jpegxl"


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


DATASET_VERSIONS: dict[DatasetKey, DatasetVersion] = {
    DatasetKey.GEORVQ: DatasetVersion(
        download_method=lambda: download_dataset(DatasetKey.GEORVQ),
        description="GeoRVQ compressed Sentinel-2 data",
    ),
    DatasetKey.JPEGXL: DatasetVersion(
        download_method=lambda: download_dataset(DatasetKey.JPEGXL),
        description="JPEG XL compressed Sentinel-2 data",
    )
}


def get_dir_size_bytes(path: Path) -> int:
    """Returns the total size of a directory in bytes.

    Args:
        path (Path): Path of the directory to analyse.

    Returns:
        int: Total size of the directory in bytes.
    """
    total = 0
    for root, _, files in path.walk():
        total += sum((root / file).stat().st_size for file in files)
    return total


def run(dataset_version: DatasetKey):
    """Run the demo, for a given dataset version.

    Args:
        dataset_version (DatasetKey): Enum item identifying the dataset to be run against.
    """
    dataset = DATASET_VERSIONS[dataset_version]

    print("#########")
    print(f"Dataset: {dataset.description}")
    print(f"Source: {REPO_ID}")
    print("Downloading dataset...")

    start = time()

    scene_path = dataset.download_method()

    time_taken = time() - start
    download_size = get_dir_size_bytes(scene_path)

    print("#########")
    print(f"Download size: {humanize.naturalsize(download_size)}")
    print(f"Download completed in {humanize.precisedelta(time_taken)}")
    print(f"Saved to: {scene_path}")


if __name__ == "__main__":
    run(dataset_version=DatasetKey.GEORVQ)
