from typing import List, Tuple


def generate_centered_array(
    target_val: float,
    n_points: int,
    majority_ratio: float,
    fluctuation_intensity: float,
) -> List[float]: ...


def generate_time_axis(n_points: int, sampling_rate: float) -> List[float]: ...


def generate_signal_with_time(
    target_val: float,
    n_points: int,
    majority_ratio: float,
    fluctuation_intensity: float,
    sampling_rate: float,
) -> Tuple[List[float], List[float]]: ...
