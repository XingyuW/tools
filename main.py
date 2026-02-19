import matplotlib.pyplot as plt
import rust_tools
import fformat


def main() -> None:
    target_value = 4.2
    n_samples = 2000
    majority_fraction = 0.90
    noise_sigma = 0.4
    sampling_rate = 100.0

    time_axis, signal_data = rust_tools.generate_signal_with_time(
        target_value,
        n_samples,
        majority_fraction,
        noise_sigma,
        sampling_rate,
    )

    # Initialize plotting parameters
    fformat.ensure_initialized()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_axis, signal_data, linewidth=1)
    ax.set_title("Signal Generated in Rust, Plotted in Python")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal Value")
    fformat.saveFig(fig, "signal_plot")


if __name__ == "__main__":
    main()
