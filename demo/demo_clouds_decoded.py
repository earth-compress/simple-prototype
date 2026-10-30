"""Code for running the Clouds Decoded-based demo."""

import os
from pathlib import Path
from time import time

import humanize
from clouds_decoded_wrapper import clouds_decoded_visualise, run_clouds_decoded

from demo import DATASET_VERSIONS, DatasetKey, get_dir_size_bytes

# Default compression level for each dataset type
COMPRESSION_LEVELS: dict[DatasetKey, str] = {
    DatasetKey.GEORVQ: "8-4",
    DatasetKey.JPEGXL: "d-0.17",
    DatasetKey.SZ3: "abs-63",
}


def run(dataset_version: DatasetKey, device: str | None = None, compression_level: str | None = None):
    """Run the demo, for a given dataset version.

    Args:
        dataset_version (DatasetKey): Enum item identifying the dataset to be run against.
        device (str | None, optional): Compute device, ('cuda', 'cpu', 'mps', or None=auto).
        compression_level (str | None, optional): Compression level for the file.
    """
    # Set the location for where CD will put its stuff
    os.environ["CLOUDS_DECODED_ASSETS_DIR"] = "data"

    dataset = DATASET_VERSIONS[dataset_version]

    if compression_level is None:
        compression_level = COMPRESSION_LEVELS[dataset_version]

    print("#########")
    print(f"Dataset: {dataset.description}")
    print(f"Compression level: {compression_level}")
    print("Downloading dataset...")

    download_start = time()

    scene_path = dataset.download_method()

    download_time_taken = time() - download_start
    download_size = get_dir_size_bytes(scene_path)

    print("#########")
    print(f"Download size: {humanize.naturalsize(download_size)}")
    print(f"Download completed in {humanize.precisedelta(download_time_taken)}")
    print(f"Saved to: {scene_path}")

    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)

    # Run the Clouds Decoded pipeline
    print("#########")
    print("Running Clouds Decoded...")

    processing_start = time()
    run_clouds_decoded(
        output_path=output_path, scene_path=scene_path, device=device, compression_level=compression_level
    )
    processing_time_taken = time() - processing_start

    # Generate visualisation
    print("Generating visualisation...")
    visualisation_start = time()
    clouds_decoded_visualise(output_path=output_path, scene_path=scene_path)
    visualisation_time_taken = time() - visualisation_start

    total_time = download_time_taken + processing_time_taken + visualisation_time_taken

    # Results
    print("#########")
    print(f"Dataset: {dataset.description}")
    print(f"Compression level: {compression_level}")
    print(f"Download size: {humanize.naturalsize(download_size)}")
    print(f"Download time: {humanize.precisedelta(download_time_taken)}")
    print(f"Processing time: {humanize.precisedelta(processing_time_taken)}")
    print(f"Visualisation time: {humanize.precisedelta(visualisation_time_taken)}")
    print(f"Total time: {humanize.precisedelta(total_time)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run(dataset_version=DatasetKey.GEORVQ, device=None)
