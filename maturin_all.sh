#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "$ROOT_DIR/.venv/bin/maturin" ]]; then
  MATURIN="$ROOT_DIR/.venv/bin/maturin"
else
  MATURIN="maturin"
fi

echo "[1/2] Building generator (signal_generator)..."
"$MATURIN" develop -m "$ROOT_DIR/rust/signal_generator/Cargo.toml" "$@"

echo "[2/2] Building font_process (font_replacer)..."
"$MATURIN" develop -m "$ROOT_DIR/rust/font_process/Cargo.toml" "$@"

echo "Done: both PyO3 extensions were built and installed."
