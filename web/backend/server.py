"""
Resonance API — FastAPI backend for GPU-accelerated acoustic simulation.

Calls the compiled acoustic_sim binary, computes metrics, returns results.
"""

import base64
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BINARY = os.environ.get("RESONANCE_BINARY", "/opt/resonance/build/acoustic_sim")
MAX_RAYS = 200_000
TIMEOUT_S = 120

app = FastAPI(title="Resonance API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MATERIAL_PRESETS = [
    "concrete", "brick", "drywall", "plaster",
    "glass", "wood_floor", "carpet_thin", "carpet_thick",
    "acoustic_foam", "acoustic_panel", "curtain", "audience",
]

ROOM_PRESETS = {
    "recording_studio": {
        "label": "Recording Studio",
        "dims": [4, 3, 2.8],
        "materials": {"floor": "carpet_thick", "walls": "acoustic_foam", "ceiling": "acoustic_panel"},
        "source": [1.5, 1.2, 1.4],
        "listener": [2.5, 1.2, 1.4],
        "rays": 80000,
    },
    "living_room": {
        "label": "Living Room",
        "dims": [6, 4, 2.6],
        "materials": {"floor": "carpet_thin", "walls": "drywall", "ceiling": "plaster"},
        "source": [1.5, 1.0, 2.0],
        "listener": [4.5, 1.0, 2.0],
        "rays": 80000,
    },
    "concert_hall": {
        "label": "Concert Hall",
        "dims": [30, 15, 12],
        "materials": {"floor": "wood_floor", "walls": "concrete", "ceiling": "acoustic_panel"},
        "source": [5, 1.5, 7.5],
        "listener": [20, 1.5, 7.5],
        "rays": 150000,
    },
    "bathroom": {
        "label": "Bathroom",
        "dims": [2.5, 2.5, 2.4],
        "materials": {"floor": "glass", "walls": "glass", "ceiling": "plaster"},
        "source": [0.8, 1.5, 1.2],
        "listener": [1.7, 1.5, 1.2],
        "rays": 50000,
    },
    "cathedral": {
        "label": "Cathedral",
        "dims": [40, 20, 18],
        "materials": {"floor": "concrete", "walls": "brick", "ceiling": "concrete"},
        "source": [10, 1.5, 10],
        "listener": [30, 1.5, 10],
        "rays": 150000,
    },
    "lecture_hall": {
        "label": "Lecture Hall",
        "dims": [15, 6, 10],
        "materials": {"floor": "carpet_thin", "walls": "drywall", "ceiling": "acoustic_panel"},
        "source": [2, 1.5, 5],
        "listener": [12, 1.5, 5],
        "rays": 100000,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_shoebox_obj(dims: list[float]) -> str:
    """Generate a closed box OBJ with named groups for each surface."""
    x, y, z = dims
    verts = [
        (0, 0, 0), (x, 0, 0), (x, 0, z), (0, 0, z),  # floor
        (0, y, 0), (x, y, 0), (x, y, z), (0, y, z),  # ceiling
    ]
    lines = []
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")

    # Groups with face indices (OBJ 1-indexed)
    groups = {
        "floor":      [(1, 2, 3), (1, 3, 4)],
        "ceiling":    [(5, 7, 6), (5, 8, 7)],
        "wall_front": [(1, 5, 6), (1, 6, 2)],
        "wall_back":  [(3, 7, 8), (3, 8, 4)],
        "wall_left":  [(1, 4, 8), (1, 8, 5)],
        "wall_right": [(2, 6, 7), (2, 7, 3)],
    }
    for gname, faces in groups.items():
        lines.append(f"g {gname}")
        for f in faces:
            lines.append(f"f {f[0]} {f[1]} {f[2]}")
    return "\n".join(lines) + "\n"


def _read_wav(path: str) -> tuple[int, np.ndarray]:
    """Read a 16-bit PCM WAV file, return (sample_rate, float32 samples)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV file")
    sr = struct.unpack_from("<I", data, 24)[0]
    bits = struct.unpack_from("<H", data, 34)[0]
    # find data chunk
    pos = 12
    while pos < len(data) - 8:
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"data":
            raw = data[pos + 8: pos + 8 + chunk_size]
            break
        pos += 8 + chunk_size
    else:
        raise ValueError("No data chunk")
    if bits == 16:
        n = len(raw) // 2
        samples = np.array(struct.unpack(f"<{n}h", raw), dtype=np.float32) / 32768.0
    elif bits == 32:
        n = len(raw) // 4
        samples = np.array(struct.unpack(f"<{n}f", raw), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported bit depth: {bits}")
    return sr, samples


def _schroeder(ir: np.ndarray) -> np.ndarray:
    power = ir.astype(np.float64) ** 2
    bw = np.cumsum(power[::-1])[::-1]
    total = bw[0]
    if total < 1e-12:
        return np.full_like(ir, -120.0, dtype=np.float64)
    return 10.0 * np.log10(bw / total + 1e-12)


def _compute_rt60(ir: np.ndarray, sr: int) -> float | None:
    curve = _schroeder(ir)
    t = np.arange(len(curve)) / sr
    i0 = np.searchsorted(-curve, 5.0)
    i1 = np.searchsorted(-curve, 65.0)
    if i1 >= len(curve) or i0 >= i1:
        return None
    slope, _ = np.polyfit(t[i0:i1], curve[i0:i1], 1)
    return None if slope >= 0 else -60.0 / slope


def _compute_edt(ir: np.ndarray, sr: int) -> float | None:
    curve = _schroeder(ir)
    t = np.arange(len(curve)) / sr
    i1 = np.searchsorted(-curve, 10.0)
    if i1 < 2:
        return None
    slope, _ = np.polyfit(t[:i1], curve[:i1], 1)
    return None if slope >= 0 else -60.0 / slope


def _compute_c50(ir: np.ndarray, sr: int) -> float:
    ir64 = ir.astype(np.float64)
    cutoff = int(0.050 * sr)
    early = np.sum(ir64[:cutoff] ** 2)
    late = np.sum(ir64[cutoff:] ** 2)
    return float(10.0 * np.log10((early + 1e-12) / (late + 1e-12)))


def _parse_debug_rays(obj_path: str, max_rays: int = 40) -> list[list[list[float]]]:
    """Parse debug_rays.obj into a list of ray paths (list of [x,y,z] points)."""
    if not os.path.exists(obj_path):
        return []
    rays: list[list[list[float]]] = []
    current: list[list[float]] = []
    verts: list[list[float]] = []
    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "l" and len(parts) >= 3:
                # line segment: l v1 v2
                i0 = int(parts[1]) - 1
                i1 = int(parts[2]) - 1
                if not current or current[-1] != verts[i0]:
                    if current:
                        rays.append(current)
                        if len(rays) >= max_rays:
                            break
                    current = [verts[i0], verts[i1]]
                else:
                    current.append(verts[i1])
    if current and len(rays) < max_rays:
        rays.append(current)
    return rays


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class SimRequest(BaseModel):
    room_type: str = "shoebox"
    dims: list[float] = Field(default=[10.0, 5.0, 8.0])
    source: list[float] = Field(default=[2.0, 1.5, 1.5])
    listener: list[float] = Field(default=[8.0, 1.5, 1.5])
    rays: int = Field(default=50000, ge=1000, le=MAX_RAYS)
    absorption: float = Field(default=0.1, ge=0.0, le=1.0)
    scattering: float = Field(default=0.1, ge=0.0, le=1.0)
    materials: dict[str, str] | None = None
    sr: int = Field(default=44100, ge=8000, le=96000)
    ir_len_ms: float = Field(default=1000.0, ge=100, le=5000)
    debug_rays: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    binary_ok = os.path.isfile(BINARY) and os.access(BINARY, os.X_OK)
    return {"status": "ok", "binary": binary_ok}


@app.get("/api/presets")
async def get_presets():
    return {"rooms": ROOM_PRESETS, "materials": MATERIAL_PRESETS}


@app.post("/api/simulate")
async def simulate(req: SimRequest):
    if not os.path.isfile(BINARY):
        raise HTTPException(503, f"Binary not found at {BINARY}")

    with tempfile.TemporaryDirectory() as tmp:
        outfile = os.path.join(tmp, "ir.wav")
        cmd = [BINARY]

        use_mesh = False
        obj_path = os.path.join(tmp, "room.obj")

        # If per-surface materials requested, generate a mesh OBJ
        if req.materials and req.room_type == "shoebox":
            obj_data = _generate_shoebox_obj(req.dims)
            with open(obj_path, "w") as f:
                f.write(obj_data)
            use_mesh = True

            cmd += ["--room", "mesh", "--mesh", obj_path]

            # Build mat-assign string — expand "walls" to all four wall groups
            WALL_GROUPS = ["wall_front", "wall_back", "wall_left", "wall_right"]
            assign_parts = []
            for surface, preset in req.materials.items():
                if preset not in MATERIAL_PRESETS:
                    continue
                if surface == "walls":
                    for wg in WALL_GROUPS:
                        assign_parts.append(f"{wg}={preset}")
                else:
                    assign_parts.append(f"{surface}={preset}")
            if assign_parts:
                cmd += ["--mat-assign", ",".join(assign_parts)]
        else:
            cmd += ["--room", req.room_type]
            if req.room_type == "dome":
                cmd += ["--dims", f"{req.dims[0]}"]
            else:
                cmd += ["--dims", ",".join(str(d) for d in req.dims)]
            cmd += ["--absorption", str(req.absorption)]

        cmd += [
            "--source", ",".join(str(s) for s in req.source),
            "--listener", ",".join(str(l) for l in req.listener),
            "--rays", str(req.rays),
            "--scattering", str(req.scattering),
            "--sr", str(req.sr),
            "--ir-len", str(req.ir_len_ms),
            "--out", outfile,
        ]

        if req.debug_rays:
            cmd += ["--debug"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "Simulation timed out")

        if result.returncode != 0:
            raise HTTPException(
                500,
                f"Simulation failed (exit {result.returncode}): {result.stderr[:500]}",
            )

        if not os.path.isfile(outfile):
            raise HTTPException(500, "No output WAV generated")

        sr, ir = _read_wav(outfile)

        # Compute metrics
        rt60 = _compute_rt60(ir, sr)
        edt = _compute_edt(ir, sr)
        c50 = _compute_c50(ir, sr)

        # Encode WAV as base64
        with open(outfile, "rb") as f:
            wav_b64 = base64.b64encode(f.read()).decode()

        # Downsample IR for waveform display (max 2000 points)
        display_len = min(len(ir), 2000)
        step = max(1, len(ir) // display_len)
        waveform = ir[::step].tolist()

        # Parse debug rays if requested
        ray_paths = []
        if req.debug_rays:
            ray_paths = _parse_debug_rays(os.path.join(tmp, "debug_rays.obj"))

        return {
            "wav_base64": wav_b64,
            "sample_rate": sr,
            "ir_length": len(ir),
            "waveform": waveform,
            "metrics": {
                "rt60": round(rt60, 3) if rt60 else None,
                "edt": round(edt, 3) if edt else None,
                "c50": round(c50, 2) if c50 else None,
            },
            "ray_paths": ray_paths,
            "stdout": result.stdout[:2000],
        }
