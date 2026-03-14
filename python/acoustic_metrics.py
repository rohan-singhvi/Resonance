"""
Acoustic quality metrics computed from a room impulse response (RIR).

All functions accept a 1D numpy array and a sample rate.
Results follow ISO 3382 definitions.
"""
import numpy as np


def schroeder_integration(ir: np.ndarray, sr: int) -> np.ndarray:
    """Backward-integrated squared impulse response, normalised to 0 dB at t=0."""
    ir = ir.astype(np.float64)
    power = ir ** 2
    backward_sum = np.cumsum(power[::-1])[::-1]
    total = backward_sum[0]
    if total < 1e-12:
        return np.full_like(ir, -120.0)
    return 10.0 * np.log10(backward_sum / total + 1e-12)


def compute_rt60(ir: np.ndarray, sr: int) -> float | None:
    """
    Reverberation time: time for energy to decay 60 dB.
    Estimated from the -5 dB to -65 dB slope on the Schroeder curve.
    Returns None if the IR is too short to reach -65 dB.
    """
    curve = schroeder_integration(ir, sr)
    t = np.arange(len(curve)) / sr

    idx_start = np.searchsorted(-curve, 5.0)
    idx_end = np.searchsorted(-curve, 65.0)

    if idx_end >= len(curve) or idx_start >= idx_end:
        return None

    slope, _ = np.polyfit(t[idx_start:idx_end], curve[idx_start:idx_end], 1)
    if slope >= 0:
        return None
    return -60.0 / slope


def compute_edt(ir: np.ndarray, sr: int) -> float | None:
    """
    Early Decay Time: decay time extrapolated from the first 10 dB of the curve.
    """
    curve = schroeder_integration(ir, sr)
    t = np.arange(len(curve)) / sr

    idx_start = 0
    idx_end = np.searchsorted(-curve, 10.0)

    if idx_end < 2:
        return None

    slope, _ = np.polyfit(t[idx_start:idx_end], curve[idx_start:idx_end], 1)
    if slope >= 0:
        return None
    return -60.0 / slope


def compute_c50(ir: np.ndarray, sr: int) -> float:
    """
    Clarity C50: ratio of early energy (0-50ms) to late energy, in dB.
    Positive values indicate good speech clarity.
    """
    ir = ir.astype(np.float64)
    cutoff = int(0.050 * sr)
    early = np.sum(ir[:cutoff] ** 2)
    late = np.sum(ir[cutoff:] ** 2)
    return 10.0 * np.log10(early / (late + 1e-12))


def print_metrics(ir: np.ndarray, sr: int) -> None:
    rt60 = compute_rt60(ir, sr)
    edt = compute_edt(ir, sr)
    c50 = compute_c50(ir, sr)

    print("--- Acoustic Metrics ---")
    print(f"RT60: {rt60:.3f} s" if rt60 is not None else "RT60: N/A (IR too short)")
    print(f"EDT:  {edt:.3f} s" if edt is not None else "EDT:  N/A")
    print(f"C50:  {c50:.2f} dB")
