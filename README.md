# Earth Compress Prototype

This is an early, fairly simple prototype to demonstrate what Earth Compress can offer in an end-to-end workflow.

---

# Running hello world

Requires uv to be installed: https://docs.astral.sh/uv/getting-started/installation/

`uv run hello`

---

# Running the demos

We have two demos available. The first downloads compressed Sentinel-2 data, while the second downloads the data and processes it using [Clouds Decoded](https://asterisk.coop/project-pages/project-clouds-decoded.html).

Both demos support GeoRVQ, JPEG XL and SZ3 compression formats.

## Download demo

This demo downloads a compressed Sentinel-2 dataset from Hugging Face. It shows the download size, download time and where the files are saved.

1. Go to the demo directory:
   ```bash
   cd demo
   ```

2. By default this will download the GeoRVQ version. Update the `dataset version` in demo.py to test a different format:

   ```bash
   if __name__ == "__main__":
      run(dataset_version=DatasetKey.JPEGXL)
   ```

   Available formats are GEORVQ, JPEGXL and SZ3.

3. Run the demo:
   ```bash
   uv run python demo.py
   ```

---

## Clouds Decoded demo

This demo downloads a compressed Sentinel-2 dataset and runs it through [Clouds Decoded](https://asterisk.coop/project-pages/project-clouds-decoded.html).

It reads the compressed data directly, then processes the imagery to generate cloud masks, cloud heights, albedo and cloud properties.

The terminal shows the download size, download time and processing time. The results are saved in the outputs directory.

1. Go to the demo directory:
   ```bash
   cd demo
   ```

2. By default, this will use the GeoRVQ version at compression level 8-4. Update the dataset_version in demo_clouds_decoded.py to test a different format:

   ```bash
   if __name__ == "__main__":
      run(
         dataset_version=DatasetKey.JPEGXL,
         device=None,
      )
   ```

   Available formats are GEORVQ, JPEGXL and SZ3.

3. You can also set a specific compression level:
   ```bash
   if __name__ == "__main__":
      run(
         dataset_version=DatasetKey.GEORVQ,
         device=None,
         compression_level="8-8",
      )
   ```

4. Install the demo dependencies:
   ```bash
   uv sync --group demo
   ```

5. Run the demo:
   ```bash
   uv run --group demo python demo_clouds_decoded.py
   ```

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
