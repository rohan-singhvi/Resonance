# Resonance — GPU-Accelerated Acoustic Simulator

A high-performance acoustic ray tracer written in **C++17** with **CUDA**, **Metal**, and **CPU** backends.

**[Try the live web tool →](https://rohansinghvi.zo.space/resonance/)**

Resonance simulates sound propagation in 3D spaces to generate realistic **Room Impulse Responses (RIR)** using stochastic ray tracing. It models frequency-dependent absorption (7 ISO octave bands), diffuse scattering, wall transmission, and air absorption. The engine automatically selects the best backend for your hardware:

* **NVIDIA GPU:** Massively parallel ray tracing via CUDA.
* **Apple GPU:** Metal compute shaders for macOS.
* **CPU:** Multi-threaded fallback via TBB + FFTW.

## Web Tool

An interactive browser-based room acoustics design tool is available at **[rohansinghvi.zo.space/resonance](https://rohansinghvi.zo.space/resonance/)**. It provides:

- 3D room visualization with Three.js (shoebox and dome geometries)
- Per-surface material assignment (14 ISO 354 material presets)
- Real-time simulation with configurable ray count, IR length, and scattering
- Waveform visualization and acoustic metrics (RT60, EDT, C50)
- Convolved audio playback — hear how a clap sounds in the simulated room
- 6 built-in room presets (studio, living room, concert hall, bathroom, cathedral, lecture hall)

The web tool is backed by a FastAPI server (`web/backend/server.py`) that calls the compiled C++ binary.

## Features

* **Three Backends:** CUDA (NVIDIA), Metal (macOS), CPU (TBB) — automatic detection
* **Frequency-Dependent Materials:**
    * 7 ISO octave bands (125 Hz – 8 kHz) with per-band absorption coefficients
    * 15 built-in material presets: concrete, brick, drywall, plaster, glass, wood floor, carpet, acoustic foam/panel, curtain, audience, and more
    * Per-surface material assignment via OBJ groups and `--mat-assign`
* **Physical Modeling:**
    * Scattering: surface roughness (specular vs. diffuse reflection)
    * Transmission: stochastic modeling of sound passing through walls
    * Air absorption: exponential attenuation with distance
    * Early/late reflection split with configurable cutoff
* **Room Geometries:** Shoebox, dome (hemisphere), and arbitrary 3D meshes (.obj)
* **SAH BVH Acceleration:** Surface Area Heuristic with 16-bin binning across all 3 axes for efficient ray-mesh intersection
* **Analysis Output:** Per-band IR WAVs, early/late split WAVs, debug ray visualization as OBJ
* **Acoustic Metrics:** RT60, EDT, C50 computed from the impulse response

## Prerequisites

You only need **Docker** installed.

* **Docker Desktop** (Mac/Windows)
* **Docker Engine** (Linux)

## Quick Start

### 1. Build the Environment

```bash
docker compose up -d --build
```

### 2. Enter the Container

```bash
docker compose exec dev bash
```

### 3. Compile the Engine

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

CMake automatically detects CUDA and Metal. If neither is available, it builds the CPU backend.

## Usage

### Basic Simulation

```bash
./acoustic_sim --room shoebox --dims 10,8,4 --out warehouse.wav
```

### Per-Surface Materials

Use an OBJ mesh with named groups for per-surface material assignment:

```bash
./acoustic_sim \
  --room mesh --mesh room.obj \
  --mat-assign "floor=carpet_thick,wall_front=drywall,wall_back=drywall,ceiling=acoustic_panel" \
  --source 2,1.5,3 --listener 8,1.5,3 \
  --rays 100000 --out studio.wav
```

### Material Presets

```bash
# Named material preset (applies to all surfaces)
./acoustic_sim --room shoebox --dims 20,15,10 --material acoustic_foam --out treated.wav

# Uniform broadband absorption
./acoustic_sim --room shoebox --dims 20,15,10 --absorption 0.05 --scattering 0.7 --out cathedral.wav
```

Available presets: `concrete`, `brick`, `drywall`, `plaster`, `glass`, `wood_panel`, `wood_floor`, `hardwood`, `carpet_thin`, `carpet_thick`, `acoustic_foam`, `acoustic_panel`, `curtain`, `audience`, `upholstered`

### Dome Room

```bash
./acoustic_sim --room dome --dims 10 --rays 50000 --out dome.wav
```

### Custom Mesh

```bash
./acoustic_sim --room mesh --mesh my_studio.obj --rays 100000 --out studio_ir.wav
```

### Full Audio Processing

Apply the room's acoustics to a dry recording:

```bash
./acoustic_sim \
  --room shoebox --dims 12,12,6 \
  --absorption 0.1 --scattering 0.5 \
  --input dry_recording.wav --mix 0.4 \
  --out wet_result.wav
```

### Debug Ray Visualization

```bash
./acoustic_sim --room shoebox --dims 10,5,8 --debug --out ir.wav
```

Generates `debug_rays.obj` with the first 100 ray paths as 3D line geometry, viewable in Blender or MeshLab.

## CLI Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--room` | `shoebox`, `dome`, `mesh` | `shoebox` |
| `--dims` | Dimensions (`L,W,H` or `Radius`) | `10,5,3` |
| `--mesh` | Path to `.obj` file (required for `mesh`) | - |
| `--source` | Source position (`x,y,z`) | `2,1.5,1.5` |
| `--listener` | Listener position (`x,y,z`) | `8,1.5,1.5` |
| `--rays` | Number of rays | `100000` |
| `--material` | Named material preset | - |
| `--absorption` | Uniform absorption (`0.0`–`1.0`) | `0.1` |
| `--scattering` | Surface roughness (`0.0`–`1.0`) | `0.1` |
| `--trans` | Transmission probability (`0.0`–`1.0`) | `0.0` |
| `--thick` | Wall thickness (meters) | `0.2` |
| `--mat-assign` | Per-group materials: `"floor=carpet,walls=concrete"` | - |
| `--sr` | Sample rate (Hz) | `44100` |
| `--ir-len` | IR duration (ms) | `1000` |
| `--air-absorption` | Air absorption coefficient per meter | `0.001` |
| `--early-ms` | Early reflection cutoff (ms) | `80` |
| `--listener-radius` | Listener sphere radius (m) | `0.5` |
| `--input` | Input `.wav` for convolution | - |
| `--mix` | Wet/dry mix (`0.0`–`1.0`) | `0.4` |
| `--out` | Output `.wav` filename | `out.wav` |
| `--debug` | Export ray paths to `debug_rays.obj` | - |

## Performance

Typical performance (100k rays, shoebox room):

| Hardware | Backend | Time |
|----------|---------|------|
| MacBook Pro M1 | CPU (TBB) | ~2–5s |
| MacBook Pro M1 | Metal | ~0.5s |
| NVIDIA RTX 4090 | CUDA | ~0.3s |
| NVIDIA H100 | CUDA | ~0.1s |

## Troubleshooting

**No rays hit the listener:**
- Increase `--rays` (try 500,000+)
- Check source/listener positions are inside the room
- For mesh mode, ensure the mesh is watertight

**IR sounds too quiet:**
- Lower `--absorption` (more reflective surfaces)
- Increase `--rays` for better statistical convergence

**Mesh simulation is slow:**
- SAH BVH is automatically built for acceleration
- GPU mode is significantly faster for complex meshes
- Reduce triangle count or simplify geometry

## License

GPL-3.0
