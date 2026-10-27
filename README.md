# Earth Compress Prototype

This is an early, fairly simple prototype to demonstrate what Earth Compress can offer in an end-to-end workflow.

---

# Running hello world

Requires uv to be installed: https://docs.astral.sh/uv/getting-started/installation/

`uv run hello`

---

# Running the demo

Go to the demo directory:
```bash
cd demo
```

In demo.py set `device=None` where run() is called:

```bash
if __name__ == "__main__":
   run(dataset_version=DatasetKey.ORIGINAL, device=None)
```

Run the demo:
```bash
uv run python demo.py
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
