from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

_native: Any = import_module("signal_generator")
generate_centered_array = cast(
    Callable[[float, int, float, float], list[float]],
    getattr(_native, "generate_centered_array"),
)
generate_time_axis = cast(
    Callable[[int, float], list[float]],
    getattr(_native, "generate_time_axis"),
)
generate_signal_with_time = cast(
    Callable[[float, int, float, float, float], tuple[list[float], list[float]]],
    getattr(_native, "generate_signal_with_time"),
)

__all__ = [
    "generate_centered_array",
    "generate_time_axis",
    "generate_signal_with_time",
]