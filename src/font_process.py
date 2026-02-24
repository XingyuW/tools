from collections.abc import Sequence
from importlib import import_module
from typing import Any, Callable, cast

_native: Any = import_module("font_process")
replace_subtitle_fonts = cast(
	Callable[[Sequence[str], str, str, Sequence[str], bool, bool], tuple[int, int]],
	getattr(_native, "replace_subtitle_fonts"),
)

__all__ = ["replace_subtitle_fonts"]