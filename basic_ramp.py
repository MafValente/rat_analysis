"""Create and visualize the sound ramp envelope."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_ramp(ramp_time: float, fs: float) -> np.ndarray:
    """Return the linear-amplitude attack ramp used by ``_apply_ramp``.

    Parameters
    ----------
    ramp_time
        Ramp duration in seconds.
    fs
        Sampling rate in Hz.
    """
    ramp_samples = int(np.floor(fs * ramp_time))
    if ramp_samples < 2:
        raise ValueError("ramp_time must contain at least two samples")

    normalized_time = np.linspace(0.0, 1.0, ramp_samples)
    warped_time = normalized_time**0.65
    return (0.5 * (1.0 - np.cos(np.pi * warped_time))) ** 2


def _apply_ramp(self) -> None:
    # Calculate sampling rate and ramp duration in samples
    fs = self.sound.shape[0] / self.sound[-1, 0]
    ramp_samples = int(np.floor(fs * self.ramp_time))

    ramp = make_ramp(self.ramp_time, fs)

    # Build the full envelope (attack, sustain, release)
    envelope = np.concatenate(
        (ramp, np.ones(self.sound.shape[0] - ramp_samples * 2), np.flip(ramp)),
        axis=None,
    )

    # Apply identically to both channels to guarantee stable ILD
    self.sound[:, 0] *= envelope
    self.sound[:, 1] *= envelope


def plot_ramps_db(
    ramp_durations_ms: tuple[int, ...] = (100, 200, 500),
    fs: float = 48_000.0,
    target_spl: float = 20.0,
    threshold_spl: float = 10.0,
    db_floor: float = -80.0,
    output_path: str | Path = "basic_ramp_envelopes_db.png",
) -> Path:
    """Plot attack ramps applied to a target sound level and save the figure.

    ``db_floor`` is the lowest displayed level relative to ``target_spl``.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    amplitude_floor = 10.0 ** (db_floor / 20.0)

    for duration_ms in ramp_durations_ms:
        duration_s = duration_ms / 1000.0
        ramp = make_ramp(duration_s, fs)
        time_ms = np.linspace(0.0, duration_ms, ramp.size)
        ramp_spl = target_spl + 20.0 * np.log10(
            np.maximum(ramp, amplitude_floor)
        )
        ax.plot(time_ms, ramp_spl, linewidth=2, label=f"{duration_ms} ms")

    ax.set(
        title=f"Ramp envelopes applied to a {target_spl:g} dB SPL sound",
        xlabel="Time from ramp onset (ms)",
        ylabel="Sound pressure level (dB SPL)",
        xlim=(0, max(ramp_durations_ms)),
        ylim=(target_spl + db_floor, target_spl + 1),
    )
    ax.axhline(
        target_spl,
        color="0.35",
        linewidth=0.8,
        linestyle="--",
        label=f"Target ({target_spl:g} dB SPL)",
    )
    ax.axhline(
        threshold_spl,
        color="0.45",
        linewidth=1.4,
        linestyle="--",
        label=f"{threshold_spl:g} dB SPL",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Ramp duration")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    saved_path = plot_ramps_db()
    print(f"Saved plot to {saved_path.resolve()}")
