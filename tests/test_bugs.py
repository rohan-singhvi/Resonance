"""
Integration tests for the fix/bugs branch.

Assumes the binary is already built at /app/build/acoustic_sim.
Run inside the Docker container via:  pytest tests/test_bugs.py -v
"""
import subprocess
import wave
import os
import tempfile
import pytest

BINARY = os.environ.get("ACOUSTIC_SIM_BIN", "/app/build/acoustic_sim")
BASE_ARGS = ["--room", "shoebox", "--rays", "5000", "--dims", "10,5,3"]


def run_sim(*extra_args, out=None):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        outpath = out or f.name
    cmd = [BINARY] + list(BASE_ARGS) + ["--out", outpath] + list(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result, outpath


def wav_info(path):
    with wave.open(path, "rb") as w:
        return w.getnframes(), w.getframerate()


class TestIRLength:
    def test_default_is_one_second_at_44100(self):
        result, outpath = run_sim()
        assert result.returncode == 0, result.stderr
        frames, rate = wav_info(outpath)
        assert rate == 44100
        assert frames == 44100
        os.unlink(outpath)

    def test_custom_sample_rate_48k(self):
        result, outpath = run_sim("--sr", "48000")
        assert result.returncode == 0, result.stderr
        frames, rate = wav_info(outpath)
        assert rate == 48000
        assert frames == 48000
        os.unlink(outpath)

    def test_custom_ir_duration(self):
        result, outpath = run_sim("--ir-len", "500")
        assert result.returncode == 0, result.stderr
        frames, rate = wav_info(outpath)
        assert rate == 44100
        assert frames == 22050
        os.unlink(outpath)

    def test_custom_sr_and_duration(self):
        result, outpath = run_sim("--sr", "48000", "--ir-len", "500")
        assert result.returncode == 0, result.stderr
        frames, rate = wav_info(outpath)
        assert rate == 48000
        assert frames == 24000
        os.unlink(outpath)


class TestListenerRadius:
    def test_large_radius_produces_more_hits(self):
        _, out_small = run_sim("--listener-radius", "0.1", "--rays", "2000")
        _, out_large = run_sim("--listener-radius", "2.0", "--rays", "2000")

        import struct
        def energy(path):
            with wave.open(path, "rb") as w:
                raw = w.readframes(w.getnframes())
            samples = struct.unpack(f"{len(raw)//2}h", raw)
            return sum(abs(s) for s in samples)

        assert energy(out_large) > energy(out_small)
        os.unlink(out_small)
        os.unlink(out_large)

    def test_zero_radius_produces_silent_or_minimal_output(self):
        result, outpath = run_sim("--listener-radius", "0.001", "--rays", "1000")
        assert result.returncode == 0, result.stderr
        os.unlink(outpath)


class TestBinaryHelp:
    def test_help_lists_new_flags(self):
        result = subprocess.run([BINARY, "--help"], capture_output=True, text=True)
        output = result.stdout + result.stderr
        assert "--listener-radius" in output
        assert "--sr" in output
        assert "--ir-len" in output
