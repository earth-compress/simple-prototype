"""Code for running the Clouds Decoded-based demo."""
import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from time import time

import humanize
from clouds_decoded.assets import download_asset
from clouds_decoded_wrapper import clouds_decoded_visualise, run_clouds_decoded
from pydantic import BaseModel


class DatasetVersion(BaseModel):
    """Model to hold information on each compression-level version of the datasets."""
    download_method: Callable[[], Path]
    description: str


class DatasetKey(StrEnum):
    """Enum holding keys for each of the dataset versions."""
    ORIGINAL = "original"

DATASET_VERSIONS: dict[DatasetKey, DatasetVersion] = {
    DatasetKey.ORIGINAL: DatasetVersion(
        download_method=lambda: download_asset("sample_scene"),
        description="Original, uncompressed Sentinel data",
    ),
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


def run(dataset_version: DatasetKey, device: str | None = None):
    """Run the demo, for a given dataset version.

    Args:
        dataset_version (DatasetKey): Enum item identifying the dataset to be run against.
        device (str | None, optional): Compute device, ('cuda', 'cpu', 'mps', or None=auto).
    """
    # Set the location for where CD will put its stuff
    os.environ["CLOUDS_DECODED_ASSETS_DIR"] = "data"

    dataset = DATASET_VERSIONS[dataset_version]

    scene_path = dataset.download_method()
    download_size = get_dir_size_bytes(scene_path)

    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)

    start = time()

    run_clouds_decoded(output_path=output_path, scene_path=scene_path, device=device)
    clouds_decoded_visualise(output_path=output_path, scene_path=scene_path)

    time_taken = time() - start

    print("#########")
    print(f"Download size: {humanize.naturalsize(download_size)}")
    print(f"Processing done in {humanize.precisedelta(time_taken)}")


if __name__ == "__main__":
    run(dataset_version=DatasetKey.ORIGINAL, device="mps")
