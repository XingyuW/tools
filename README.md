# tools

Rust-powered data generation exposed to Python through `PyO3` and `maturin`.

## Build and install the Rust Python module

```bash
uv add maturin
uv run maturin develop
```

This builds and installs the Python extension module `rust_tools` into your active environment.

## Build both PyO3 modules together

`maturin develop` only supports one Cargo manifest per invocation. To build both modules in one command from repo root:

```bash
./maturin_develop_all.sh
```

Pass-through flags are supported (for example release mode):

```bash
./maturin_develop_all.sh --release
```

## Build and use subtitle font replacer (`font_process`)

```bash
cd rust/font_process
uv run maturin develop
```

This builds and installs the Python extension module `subtitle_font_replace` into your active environment.

Example usage:

```python
import subtitle_font_replace

scanned, modified = subtitle_font_replace.replace_subtitle_fonts(
    from_fonts=["Arial", "MS UI Gothic"],
    to_font="Sarasa Mono SC",
    root_dir="/path/to/subtitles",
    file_types=["ass", ".ssa"],
    backup=True,
    dry_run=False,
)

print(scanned, modified)
```

Parameters:

- `from_fonts`: font names to replace from
- `to_font`: destination font name
- `root_dir`: folder to recursively scan
- `file_types`: subtitle extensions to process (with or without leading `.`)
- `backup` (optional, default `True`): create `.bak` backup files
- `dry_run` (optional, default `False`): detect changes without writing files

## Run the Python plot

```bash
uv run python main.py
```

`main.py` calls Rust functions to generate the signal and time axis, then plots them in Python using `matplotlib`.
