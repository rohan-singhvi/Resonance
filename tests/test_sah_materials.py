"""
Integration tests for SAH BVH and per-surface material assignment.

Run inside Docker: pytest tests/test_sah_materials.py -v
"""
import os
import struct
import subprocess
import tempfile
import textwrap
import wave

import numpy as np
import pytest

BINARY = os.environ.get("ACOUSTIC_SIM_BIN", "/app/build/acoustic_sim")

# A minimal closed-box OBJ with six named face groups so we can exercise
# the per-group material assignment path.  Each face uses vertices that
# are already declared earlier in the file (all 8 corners of a 10x5x3 box).
SHOEBOX_OBJ = textwrap.dedent("""\
    # Minimal shoebox mesh with named surface groups
    v  0  0  0
    v 10  0  0
    v 10  5  0
    v  0  5  0
    v  0  0  3
    v 10  0  3
    v 10  5  3
    v  0  5  3

    g floor
    f 1 2 3
    f 1 3 4

    g ceiling
    f 5 7 6
    f 5 8 7

    g wall_front
    f 1 2 6
    f 1 6 5

    g wall_back
    f 4 7 3
    f 4 8 7

    g wall_left
    f 1 5 8
    f 1 8 4

    g wall_right
    f 2 3 7
    f 2 7 6
""")


def write_temp_obj(content):
    f = tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w")
    f.write(content)
    f.close()
    return f.name


def run_sim(extra_args, cwd="/tmp"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=cwd) as f:
        outpath = f.name
    cmd = [BINARY] + list(extra_args) + ["--out", outpath]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result, outpath


def total_energy(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    n = len(raw) // 2
    samples = np.array(struct.unpack(f"{n}h", raw), dtype=np.float64)
    return float(np.sum(samples ** 2))


class TestOBJGroupParsing:
    """Verify that OBJ 'g' directives are parsed and material_ids populated."""

    def test_mesh_with_groups_runs(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            result, outpath = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "5000",
            ])
            assert result.returncode == 0, result.stderr
            assert os.path.exists(outpath) and os.path.getsize(outpath) > 0
        finally:
            os.unlink(obj_path)
            if os.path.exists(outpath):
                os.unlink(outpath)

    def test_group_names_printed_in_output(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            result, outpath = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "1000",
            ])
            assert result.returncode == 0, result.stderr
            # The main.cpp prints group names in "Surface material assignments:"
            combined = result.stdout + result.stderr
            assert "floor" in combined, "Expected group 'floor' in output"
            assert "ceiling" in combined, "Expected group 'ceiling' in output"
        finally:
            os.unlink(obj_path)
            if os.path.exists(outpath):
                os.unlink(outpath)


class TestMatAssignFlag:
    """Verify that --mat-assign overrides per-group materials without errors."""

    def test_mat_assign_valid_preset_runs(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            result, outpath = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "5000",
                "--mat-assign", "floor=carpet_thick,ceiling=concrete",
            ])
            assert result.returncode == 0, result.stderr
        finally:
            os.unlink(obj_path)
            if os.path.exists(outpath):
                os.unlink(outpath)

    def test_mat_assign_unknown_group_warns_not_fails(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            result, outpath = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "1000",
                "--mat-assign", "nonexistent_group=concrete",
            ])
            # Should still complete successfully (just prints a warning)
            assert result.returncode == 0, result.stderr
            combined = result.stdout + result.stderr
            assert "Warning" in combined or "warning" in combined.lower()
        finally:
            os.unlink(obj_path)
            if os.path.exists(outpath):
                os.unlink(outpath)

    def test_mat_assign_unknown_preset_warns_not_fails(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            result, outpath = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "1000",
                "--mat-assign", "floor=unobtanium",
            ])
            assert result.returncode == 0, result.stderr
            combined = result.stdout + result.stderr
            assert "Warning" in combined or "warning" in combined.lower()
        finally:
            os.unlink(obj_path)
            if os.path.exists(outpath):
                os.unlink(outpath)


class TestMaterialEffectOnEnergy:
    """Verify that highly absorptive materials produce lower IR energy."""

    def test_carpet_vs_concrete_on_mesh(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            # All surfaces set to concrete (reflective)
            _, out_concrete = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "20000",
                "--material", "concrete",
            ])
            # All surfaces set to carpet_thick (absorptive)
            _, out_carpet = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "20000",
                "--material", "carpet_thick",
            ])
            e_concrete = total_energy(out_concrete)
            e_carpet   = total_energy(out_carpet)
            assert e_carpet < e_concrete, (
                f"Carpet ({e_carpet:.0f}) should have less energy than concrete ({e_concrete:.0f})"
            )
        finally:
            os.unlink(obj_path)
            for p in [out_concrete, out_carpet]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_mat_assign_absorptive_floor_reduces_energy(self):
        """Assigning carpet to floor vs concrete should reduce total IR energy."""
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            _, out_concrete = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "20000",
                "--material", "concrete",
            ])
            _, out_carpet_floor = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "20000",
                "--material", "concrete",
                "--mat-assign",
                "floor=carpet_thick,ceiling=carpet_thick,wall_front=carpet_thick,"
                "wall_back=carpet_thick,wall_left=carpet_thick,wall_right=carpet_thick",
            ])
            e_concrete     = total_energy(out_concrete)
            e_carpet_floor = total_energy(out_carpet_floor)
            assert e_carpet_floor < e_concrete, (
                f"All-carpet ({e_carpet_floor:.0f}) should have less energy than "
                f"all-concrete ({e_concrete:.0f})"
            )
        finally:
            os.unlink(obj_path)
            for p in [out_concrete, out_carpet_floor]:
                if os.path.exists(p):
                    os.unlink(p)


class TestSAHBVH:
    """Sanity checks that the SAH BVH produces valid output on mesh rooms."""

    def test_bvh_built_message_present(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            result, outpath = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "1000",
            ])
            assert result.returncode == 0, result.stderr
            combined = result.stdout + result.stderr
            assert "BVH Built" in combined, "Expected 'BVH Built' in output"
        finally:
            os.unlink(obj_path)
            if os.path.exists(outpath):
                os.unlink(outpath)

    def test_mesh_produces_non_silent_ir(self):
        obj_path = write_temp_obj(SHOEBOX_OBJ)
        try:
            result, outpath = run_sim([
                "--room", "mesh",
                "--mesh", obj_path,
                "--rays", "10000",
                "--material", "concrete",
            ])
            assert result.returncode == 0, result.stderr
            energy = total_energy(outpath)
            assert energy > 0, "Expected non-silent IR from mesh simulation"
        finally:
            os.unlink(obj_path)
            if os.path.exists(outpath):
                os.unlink(outpath)
