# Earth Compress Prototype

This is an early, fairly simple prototype to demonstrate what Earth Compress can offer in an end-to-end workflow.

The **ec_prototype** package provides a clean interface for downloading and decoding datasets compressed under different sample algorithms.

A notebook is also provided to demonstrate this in use.

---

# Installing the package

`pip install https://github.com/earth-compress/simple-prototype.git`

Example usage

```
from ec_prototype.datasets import DATASETS, DatasetKey
dataset = DATASETS[DatasetKey.GEORVQ]
dataset.download() # Download the data
dataset.decode()
```

---

# Running the notebook

Requires uv to be installed: https://docs.astral.sh/uv/getting-started/installation/

To install dependencies necessary dependencies run `uv synv --group demo`, this will create a virtual environment which should then be used for the kernel of the notebook.

This demo notebook demonstrates the use of this ec_prototype package to download and decode compressed Sentinel-2 data and then display the content visually. The GeoRVQ and JPEG XL compression formats are currently supported.

---

# Linting and Formatting

This project uses Ruff for linting and formatting.

Run the linter with:

```bash
uv run ruff check
```

Run fixes highlighted by the linter with:

```bash
uv run ruff check --fix
```

Run the formatter with:
```bash
uv run ruff format
```
