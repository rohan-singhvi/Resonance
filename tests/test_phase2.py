"""
Integration tests for feat/phase-2: frequency-dependent materials.

Run inside Docker: pytest tests/test_phase2.py -v
"""
import subprocess
import wave
import os
import struct
import tempfile
import numpy as np
import pytest

BINARY = os.environ.get("ACOUSTIC_SIM_BIN", "/app/build/acoustic_sim")
BASE_ARGS = ["--room", "shoebox", "--rays", "10000", "--dims", "10,5,3"]
BAND_NAMES = ["125hz", "250hz", "500hz", "1khz", "2khz", "4khz", "8khz"]


def run_sim(*extra_args, cwd="/tmp"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=cwd) as f:
        outpath = f.name
    cmd = [BINARY] + list(BASE_ARGS) + ["--out", outpath] + list(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result, outpath


def band_path(cwd, band):
    return os.path.join(cwd, f"out_band_{band}.wav")


def total_energy(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    n = len(raw) // 2
    samples = np.array(struct.unpack(f"{n}h", raw), dtype=np.float64)
    return np.sum(samples ** 2)


class TestBandFiles:
    def test_all_band_files_created(self):
        result, outpath = run_sim(cwd="/tmp")
        assert result.returncode == 0, result.stderr
        for band in BAND_NAMES:
            assert os.path.exists(band_path("/tmp", band)), f"Missing out_band_{band}.wav"
        os.unlink(outpath)

    def test_band_files_correct_length(self):
        result, outpath = run_sim("--sr", "48000", "--ir-len", "500", cwd="/tmp")
        assert result.returncode == 0, result.stderr
        for band in BAND_NAMES:
            p = band_path("/tmp", band)
            with wave.open(p, "rb") as w:
                assert w.getnframes() == 24000, f"{band}: expected 24000 frames"
                assert w.getframerate() == 48000
        os.unlink(outpath)


class TestMaterialPresets:
    def test_known_material_runs(self):
        for mat in ["concrete", "carpet_thick", "glass", "acoustic_foam", "audience"]:
            result, outpath = run_sim("--material", mat, cwd="/tmp")
            assert result.returncode == 0, f"Material {mat} failed: {result.stderr}"
            os.unlink(outpath)

    def test_unknown_material_exits_nonzero(self):
        result, outpath = run_sim("--material", "unobtanium", cwd="/tmp")
        assert result.returncode != 0
        if os.path.exists(outpath):
            os.unlink(outpath)

    def test_absorptive_material_has_less_energy_than_reflective(self):
        _, out_concrete = run_sim("--material", "concrete", "--rays", "20000", cwd="/tmp")
        _, out_foam = run_sim("--material", "acoustic_foam", "--rays", "20000", cwd="/tmp")

        e_concrete = total_energy(out_concrete)
        e_foam = total_energy(out_foam)

        assert e_foam < e_concrete, (
            f"Foam ({e_foam:.0f}) should have less energy than concrete ({e_concrete:.0f})"
        )
        os.unlink(out_concrete)
        os.unlink(out_foam)

    def test_backward_compat_absorption_flag(self):
        result, outpath = run_sim("--absorption", "0.5", cwd="/tmp")
        assert result.returncode == 0, result.stderr
        os.unlink(outpath)


class TestFrequencyDependence:
    def test_carpet_attenuates_high_bands_more_than_low(self):
        _, out = run_sim("--material", "carpet_thick", "--rays", "20000", cwd="/tmp")
        assert os.path.exists(out)

        e_low = total_energy(band_path("/tmp", "125hz"))
        e_high = total_energy(band_path("/tmp", "8khz"))

        assert e_high < e_low, (
            f"Carpet should absorb more at 8kHz ({e_high:.0f}) than 125Hz ({e_low:.0f})"
        )
        os.unlink(out)

    def test_glass_attenuates_low_bands_more_than_high(self):
        _, out = run_sim("--material", "glass", "--rays", "20000", cwd="/tmp")
        assert os.path.exists(out)

        e_low = total_energy(band_path("/tmp", "125hz"))
        e_high = total_energy(band_path("/tmp", "4khz"))

        assert e_low < e_high, (
            f"Glass should absorb more at 125Hz ({e_low:.0f}) than 4kHz ({e_high:.0f})"
        )
        os.unlink(out)
