import os
import platform
import subprocess
import struct
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY    = os.path.join(REPO_ROOT, "build", "acoustic_sim")


def is_metal_build():
    """Return True only on macOS with a binary that was built with ENABLE_METAL."""
    if platform.system() != "Darwin":
        return False
    if not os.path.isfile(BINARY):
        return False
    # Check that the binary actually links Metal (quick grep of load commands).
    result = subprocess.run(
        ["otool", "-L", BINARY],
        capture_output=True, text=True
    )
    return "Metal" in result.stdout


def run_sim(*extra_args, timeout=60):
    cmd = [BINARY] + list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def read_wav(path):
    """Return (num_frames, sample_rate, samples_as_floats) for a 16-bit PCM WAV."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[0:4] == b"RIFF", f"Not a RIFF file: {path}"
    sr         = struct.unpack_from("<I", data, 24)[0]
    n_channels = struct.unpack_from("<H", data, 22)[0]
    pos = 12
    while pos < len(data) - 8:
        chunk_id   = data[pos:pos+4]
        chunk_size = struct.unpack_from("<I", data, pos+4)[0]
        if chunk_id == b"data":
            raw       = data[pos+8 : pos+8+chunk_size]
            n_samples = len(raw) // 2
            n_frames  = n_samples // n_channels
            samples   = [s / 32768.0 for s in struct.unpack(f"<{n_samples}h", raw[:n_samples*2])]
            return n_frames, sr, samples
        pos += 8 + chunk_size
    raise ValueError(f"No data chunk found in {path}")


def rms(samples):
    """Root-mean-square energy of a sample list."""
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


@unittest.skipUnless(is_metal_build(), "Metal binary not available (macOS + ENABLE_METAL required)")
class TestMetalBackend(unittest.TestCase):

    def test_backend_line_in_stdout(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            r = run_sim("--rays", "2000", "--out", out)
            self.assertIn("Metal GPU", r.stdout, "Expected 'Metal GPU' in output")
        finally:
            os.unlink(out)

    def test_shoebox_produces_non_silent_ir(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            r = run_sim("--rays", "10000", "--room", "shoebox", "--dims", "10,5,3", "--out", out)
            self.assertEqual(r.returncode, 0)
            _, _, samples = read_wav(out)
            self.assertGreater(rms(samples), 0.0, "IR should have non-zero energy")
        finally:
            os.unlink(out)

    def test_dome_produces_non_silent_ir(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            r = run_sim("--rays", "10000", "--room", "dome", "--dims", "8,0,0", "--out", out)
            self.assertEqual(r.returncode, 0)
            _, _, samples = read_wav(out)
            self.assertGreater(rms(samples), 0.0)
        finally:
            os.unlink(out)

    def test_ir_length_matches_sample_rate(self):
        """1000 ms IR at 44100 Hz should be exactly 44100 frames."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            run_sim("--rays", "5000", "--sr", "44100", "--ir-len", "1000", "--out", out)
            n, sr, _ = read_wav(out)
            self.assertEqual(sr, 44100)
            self.assertEqual(n, 44100)
        finally:
            os.unlink(out)

    def test_ir_length_custom_duration(self):
        """500 ms at 44100 Hz should produce 22050 frames."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            run_sim("--rays", "5000", "--sr", "44100", "--ir-len", "500", "--out", out)
            n, _, _ = read_wav(out)
            self.assertEqual(n, 22050)
        finally:
            os.unlink(out)

    def test_high_absorption_reduces_energy(self):
        """Low absorption walls should sustain more energy over the IR than high absorption."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_low = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_high = f.name
        try:
            run_sim("--rays", "30000", "--absorption", "0.05", "--out", out_low)
            run_sim("--rays", "30000", "--absorption", "0.95", "--out", out_high)
            _, _, s_low  = read_wav(out_low)
            _, _, s_high = read_wav(out_high)
            # After write_wav normalises both to the same peak, the high-absorption IR
            # will have almost all energy in a few early samples (sparse → lower RMS).
            # The low-absorption IR has energy spread across many more samples (higher RMS).
            self.assertGreater(rms(s_low), rms(s_high),
                "Low absorption should have higher RMS than high absorption")
        finally:
            os.unlink(out_low)
            os.unlink(out_high)

    def test_larger_listener_radius_more_energy(self):
        """A bigger listener sphere should catch more rays → more non-zero IR samples."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_small = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_large = f.name
        try:
            run_sim("--rays", "30000", "--listener-radius", "0.1", "--out", out_small)
            run_sim("--rays", "30000", "--listener-radius", "2.0", "--out", out_large)
            _, _, s_small = read_wav(out_small)
            _, _, s_large = read_wav(out_large)
            self.assertGreater(rms(s_large), rms(s_small))
        finally:
            os.unlink(out_small)
            os.unlink(out_large)

    def test_mesh_mode_with_obj(self):
        """Mesh mode should load Room.obj and produce a non-silent IR."""
        obj_path = os.path.join(REPO_ROOT, "Room.obj")
        if not os.path.isfile(obj_path):
            self.skipTest("Room.obj not found in repo root")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            r = run_sim("--rays", "5000", "--room", "mesh", "--mesh", obj_path, "--out", out)
            self.assertEqual(r.returncode, 0)
            _, _, samples = read_wav(out)
            self.assertGreater(rms(samples), 0.0)
        finally:
            os.unlink(out)


if __name__ == "__main__":
    unittest.main()
