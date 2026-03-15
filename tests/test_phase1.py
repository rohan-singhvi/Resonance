"""
Integration tests for feat/phase-1.

Tests air absorption, early/late IR splitting, and acoustic metrics.
Run inside Docker: pytest tests/test_phase1.py -v
"""
import subprocess
import wave
import os
import struct
import tempfile
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from acoustic_metrics import (
    schroeder_integration,
    compute_rt60,
    compute_edt,
    compute_c50,
)

BINARY = os.environ.get("ACOUSTIC_SIM_BIN", "/app/build/acoustic_sim")
BASE_ARGS = ["--room", "shoebox", "--rays", "10000", "--dims", "10,5,3"]


def run_sim(*extra_args):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as f:
        outpath = f.name
    cmd = [BINARY] + list(BASE_ARGS) + ["--out", outpath] + list(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
    return result, outpath


def read_wav_samples(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
        n = len(raw) // 2
    return np.array(struct.unpack(f"{n}h", raw), dtype=np.float32)


class TestEarlyLateFiles:
    def test_early_and_late_files_are_created(self):
        result, outpath = run_sim()
        assert result.returncode == 0, result.stderr
        assert os.path.exists("/tmp/out_early.wav"), "out_early.wav not created"
        assert os.path.exists("/tmp/out_late.wav"), "out_late.wav not created"
        os.unlink(outpath)

    def test_early_and_late_are_same_length_as_full(self):
        result, outpath = run_sim()
        assert result.returncode == 0, result.stderr

        with wave.open(outpath, "rb") as w:
            full_frames = w.getnframes()
        with wave.open("/tmp/out_early.wav", "rb") as w:
            early_frames = w.getnframes()
        with wave.open("/tmp/out_late.wav", "rb") as w:
            late_frames = w.getnframes()

        assert early_frames == full_frames
        assert late_frames == full_frames
        os.unlink(outpath)

    def test_early_and_late_both_have_content(self):
        result, outpath = run_sim("--rays", "20000")
        assert result.returncode == 0, result.stderr

        early = read_wav_samples("/tmp/out_early.wav")
        late = read_wav_samples("/tmp/out_late.wav")

        assert np.any(early != 0), "Early IR is completely silent"
        assert np.any(late != 0), "Late IR is completely silent"
        os.unlink(outpath)

    def test_early_file_has_correct_sample_rate(self):
        result, _ = run_sim("--sr", "48000")
        assert result.returncode == 0, result.stderr
        with wave.open("/tmp/out_early.wav", "rb") as w:
            assert w.getframerate() == 48000

    def test_custom_early_cutoff(self):
        result, outpath = run_sim("--early-ms", "40")
        assert result.returncode == 0, result.stderr
        os.unlink(outpath)


class TestAirAbsorption:
    def test_high_absorption_reduces_energy(self):
        _, out_low = run_sim("--air-absorption", "0.0")
        _, out_high = run_sim("--air-absorption", "0.1")

        energy_low = np.sum(read_wav_samples(out_low).astype(np.float64) ** 2)
        energy_high = np.sum(read_wav_samples(out_high).astype(np.float64) ** 2)

        assert energy_high < energy_low, "Higher air absorption should reduce total energy"
        os.unlink(out_low)
        os.unlink(out_high)

    def test_zero_absorption_runs_clean(self):
        result, outpath = run_sim("--air-absorption", "0.0")
        assert result.returncode == 0, result.stderr
        os.unlink(outpath)


class TestAcousticMetrics:
    def _make_ir(self, rt60_seconds=0.5, sr=44100, duration=1.0):
        n = int(sr * duration)
        t = np.arange(n) / sr
        decay_rate = np.log(1000) / rt60_seconds
        ir = np.exp(-decay_rate * t) * np.random.randn(n)
        ir[0] = 1.0
        return ir.astype(np.float32)

    def test_schroeder_starts_at_zero(self):
        ir = self._make_ir()
        curve = schroeder_integration(ir, 44100)
        assert abs(curve[0]) < 0.1

    def test_schroeder_is_monotonically_decreasing(self):
        ir = self._make_ir()
        curve = schroeder_integration(ir, 44100)
        diffs = np.diff(curve)
        assert np.all(diffs <= 0.01), "Schroeder curve should be non-increasing"

    def test_rt60_estimate_is_close(self):
        for target_rt60 in [0.3, 0.5, 0.8]:
            ir = self._make_ir(rt60_seconds=target_rt60, duration=2.0)
            estimated = compute_rt60(ir, 44100)
            assert estimated is not None, f"RT60 returned None for target {target_rt60}"
            assert abs(estimated - target_rt60) < target_rt60 * 0.2, (
                f"RT60 estimate {estimated:.3f} too far from target {target_rt60}"
            )

    def test_rt60_returns_none_for_short_ir(self):
        ir = np.ones(100, dtype=np.float32)
        result = compute_rt60(ir, 44100)
        assert result is None

    def test_c50_positive_for_dry_signal(self):
        sr = 44100
        ir = np.zeros(sr, dtype=np.float32)
        ir[0] = 1.0
        c50 = compute_c50(ir, sr)
        assert c50 > 0

    def test_c50_negative_for_late_heavy_ir(self):
        sr = 44100
        ir = np.zeros(sr, dtype=np.float32)
        cutoff = int(0.05 * sr)
        ir[cutoff + 100:] = 1.0
        c50 = compute_c50(ir, sr)
        assert c50 < 0

    def test_edt_plausible_for_synthetic_ir(self):
        ir = self._make_ir(rt60_seconds=0.5, duration=1.0)
        edt = compute_edt(ir, 44100)
        assert edt is not None
        assert 0.05 < edt < 5.0
