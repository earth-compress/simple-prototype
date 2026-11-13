<p>
  <img src="assets/earth_compress_logo.svg" alt="Earth Compress Logo" width="400">
</p>

# Earth Compress Prototype

This is an early, fairly simple prototype to demonstrate what Earth Compress can offer in an end-to-end workflow.

The **ec_prototype** package provides a clean interface for downloading and decoding datasets compressed under different sample algorithms.

A notebook is also provided to demonstrate this in use.

---

## What is Earth Compress?

Earth Compress is a three-year project funded by the [Advanced Research and Invention Agency (ARIA)](https://aria.org.uk/), led by the [National Oceanography Centre](https://www.noc.ac.uk/) and [Asterisk Labs](https://asterisk.coop/) to 'put petabytes of environmental data in everyone's pocket'. 

The project will create AI software tools and infrastructure that enable any user, from a student on a laptop to a national meteorological agency, to work with entire environmental archives that currently require enormous computing resources and budgets. 

To hear more about the project, read [ARIA’s announcement](https://aria.org.uk/insights/2026/doubling-down-on-earth-scale-data).

---

# Installing the package

Install the ec_prototype directly from GitHub using pip:

`pip install pip install git+https://github.com/earth-compress/simple-prototype.git`

## Example usage

This package demonstrates downloading and decoding compressed Sentinel-2 SAFE files.

```
# Select the compression format, GEORVQ or JPEGXL:
dataset = DATASETS[DatasetKey.GEORVQ]

# Download the compressed Sentinel-2 data
dataset.download()

# Decode the data
dataset.decode()
```
The decoded image is returned as NumPy arrays grouped by spatial resolution ("10m", "20m" and "60m").

For further details, see the [notebook](test.ipynb) file at the root of this repository.

---

# Running the notebook

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

To install necessary dependencies run: 
```
uv sync --group demo
```
This will create a virtual environment which should then be used for the kernel of the notebook.

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

---

# Project partners

The Earth Compress project is developed in collaboration with:

<p>
  <a href="https://aria.org.uk/">
    <img src="assets/aria_logo.svg"
         alt="Advanced Research and Invention Agency (ARIA)" height="60">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://noc.ac.uk">
    <img src="assets/noc_logo.svg"
         alt="The National Oceanography Centre" height="60">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://asterisk.coop/">
    <img src="assets/asterisk_labs_logo.svg"
         alt="Asterisk Labs" height="60">
  </a>
</p>
