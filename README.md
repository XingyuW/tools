# tools

Rust-powered data generation exposed to Python through `PyO3` and `maturin`.

## Build and install the Rust Python module

```bash
uv add maturin
uv run maturin develop
```

This builds and installs the Python extension module `rust_tools` into your active environment.

## Run the Python plot

```bash
uv run python main.py
```

`main.py` calls Rust functions to generate the signal and time axis, then plots them in Python using `matplotlib`.
