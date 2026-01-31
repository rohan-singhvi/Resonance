# GPU-Accelerated Acoustic Simulator (C++/CUDA)

A high-performance, hybrid acoustic ray tracer written in **C++17** and **CUDA**.

This engine simulates sound propagation in 3D spaces to generate realistic **Room Impulse Responses (RIR)**. It features a **Stochastic Ray Tracing** engine that models physical phenomena like diffuse reflection (scattering), wall transmission (insulation), and absorption. It features a hybrid architecture that automatically detects your hardware:

* **NVIDIA GPU:** Runs massively parallel ray tracing using CUDA.
* **CPU (Mac/Linux):** Falls back to multi-threaded TBB + FFTW for compatibility.

It simulates how sound bounces around a room (shoebox, dome, or a complex 3D model) to generate a "Room Impulse Response" (IR). This IR captures the acoustic "fingerprint" of the room, which is then applied to a dry audio signal (convolution) to make it sound like it was recorded in that space.

## Features

* **Hybrid Core:** Seamlessly runs on MacBook Pro (CPU) or Cloud Servers (H100/A100 GPUs).
* **Material Physics:**
    * **Absorption:** Control energy loss per bounce (e.g., Concrete vs. Foam).
    * **Scattering:** Simulate surface roughness (Specular vs. Diffuse reflection).
    * **Transmission:** Stochastic modeling of sound passing through walls (Russian Roulette).
* **Room Shapes:** Shoebox, Dome (Hemisphere), and Arbitrary 3D Meshes (.obj).
* **BVH Acceleration:** Efficient ray-mesh intersection using Bounding Volume Hierarchy.
* **Debug Visualization:** Export ray paths as OBJ files to visualize acoustic behavior.
* **Analysis Tools:** Built-in Python scripts for audio generation and spectral visualization.

## Prerequisites

You only need **Docker** installed.

* **Docker Desktop** (Mac/Windows)
* **Docker Engine** (Linux)

## Quick Start

### 1. Build the Environment

The Docker setup handles all C++ toolchains (CMake, NVCC, FFTW) and Python analysis dependencies (NumPy, Matplotlib, SoundFile, Scipy, Trimesh).

```bash
docker compose up -d --build
```

### 2. Enter the Container

All build and run commands must be executed inside the container.

```bash
docker compose exec dev bash
```

### 3. Compile the Engine

Once inside the container, build the C++ project:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

*Note: CMake will automatically detect if you have an NVIDIA GPU. If not, it will configure the project for CPU mode.*

## Usage

The executable is located at `./acoustic_sim` inside the `build/` directory.

### Basic Simulation (Shoebox)

Generate a 1-second Impulse Response for a 10m x 8m x 4m room.

```bash
./acoustic_sim --room shoebox --dims 10,8,4 --out warehouse.wav
```

### Material Physics Examples

You can tweak the material properties to simulate different environments.

**1. The "Cathedral" (Reflective, Rough/Diffuse)**
Low absorption, high scattering.

```bash
./acoustic_sim \
  --room shoebox --dims 20,15,10 \
  --absorption 0.05 \
  --scattering 0.7 \
  --out cathedral.wav
```

**2. The "Paper House" (High Transmission)**
Sound leaks through thin walls (Transmission > 0).

```bash
./acoustic_sim \
  --room shoebox --dims 5,5,3 \
  --trans 0.5 --thick 0.05 \
  --out thin_walls.wav
```

### Custom Mesh Simulation

You can use any `.obj` file. Place the file in your project root.

```bash
# Note: ../my_studio.obj because we are in the build/ folder
./acoustic_sim \
  --room mesh \
  --mesh ../my_studio.obj \
  --rays 100000 \
  --absorption 0.2 \
  --out studio_ir.wav
```

### Full Audio Processing (Reverb)

Apply the room's acoustics to a sound file immediately using `--input` and `--mix`.

```bash
./acoustic_sim \
  --room shoebox --dims 12,12,6 \
  --absorption 0.1 --scattering 0.5 \
  --input ../dry_recording.wav \
  --mix 0.4 \
  --out wet_result.wav
```

### Debug Ray Visualization

Visualize how rays bounce around your room by exporting them as 3D line geometry. The first 100 rays are recorded and saved to `debug_rays.obj`, which you can open in Blender, MeshLab, or any 3D viewer.

```bash
./acoustic_sim \
  --room dome --dims 10,10,10 \
  --rays 50000 \
  --debug \
  --out dome_ir.wav
```

This generates `debug_rays.obj` showing the actual ray paths. Each line segment represents a reflection, allowing you to verify:
- Source and listener positions
- Reflection patterns (specular vs. diffuse)
- Ray density and coverage
- Early reflections vs. late reverb

## Python Analysis Tools

The container includes scripts to generate test audio and visualize the physics.

**1. Generate Test Audio**
Creates a sparse techno beat (`techno_dry.wav`) designed for testing reverb tails.

```bash
python3 ../generate_beat.py
```

**2. Visualize Acoustics**
Generates a 3-panel analysis plot (Waveform, Spectrogram, Energy Decay) to verify your physics settings.

```bash
# Compare the dry input against your simulation output
python3 ../visualize.py techno_dry.wav wet_result.wav
```

*Output: `acoustic_analysis.png`*

**3. Generate a Test Mesh (Icosphere)**
If you don't have a 3D model, run this one-liner to create a sphere mesh:
```bash
python3 -c "import trimesh; trimesh.creation.icosphere(radius=8).export('../test_sphere.obj')"
```

## Technical Details

### Ray Tracing Engine
- **Algorithm:** Stochastic ray tracing with up to 50 bounces per ray
- **Physics:** Möller-Trumbore triangle intersection, cosine-weighted hemisphere sampling
- **Acceleration:** BVH (Bounding Volume Hierarchy) for mesh mode
- **Listener:** 0.5m radius sphere for ray capture
- **IR Length:** 1 second @ 44.1kHz (44,100 samples)

### Material Model
Each surface has four properties:
- **Absorption** (0-1): Energy absorbed per reflection (1 - reflectivity)
- **Scattering** (0-1): Mix between specular (mirror) and diffuse (Lambertian) reflection
- **Transmission** (0-1): Probability of passing through the surface (Russian Roulette)
- **Thickness** (meters): Distance rays travel when transmitted through walls

### Convolution
- **CPU:** FFTW3 for FFT-based convolution (O(n log n))
- **GPU:** cuFFT for GPU-accelerated convolution
- Automatic fallback to naive convolution if FFTW unavailable

## CLI Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--room` | `shoebox`, `dome`, `mesh` | `shoebox` |
| `--dims` | Dimensions (`L,W,H` or `Radius`) | `10,5,3` |
| `--mesh` | Path to `.obj` file (Required for `mesh` mode) | - |
| `--rays` | Number of rays to trace (More = Higher Quality) | `100000` |
| `--input` | Path to input `.wav` for reverb processing | - |
| `--mix` | Wet/Dry mix (`0.0` to `1.0`) | `0.4` |
| `--out` | Output `.wav` filename | `out.wav` |
| `--absorption` | Wall energy loss (`0.0`=Mirror, `1.0`=Dead) | `0.1` |
| `--scattering` | Surface roughness (`0.0`=Smooth, `1.0`=Diffuse) | `0.1` |
| `--trans` | Transmission probability (`0.0`=Opaque, `1.0`=Clear) | `0.0` |
| `--thick` | Wall thickness (meters) for transmission calculation | `0.2` |
| `--debug` | Export first 100 ray paths to `debug_rays.obj` | - |

## Troubleshooting

**No rays hit the listener:**
- Increase `--rays` (try 500,000+)
- Check source/listener positions are inside the room
- For mesh mode, ensure the mesh is closed (watertight)

**IR sounds too quiet:**
- Lower `--absorption` (more reflective surfaces)
- Increase `--rays` for better statistical convergence
- Check `--mix` isn't too low

**Mesh simulation is slow:**
- BVH is automatically built for acceleration
- GPU mode is significantly faster for complex meshes
- Reduce triangle count or simplify geometry

## Performance

Typical performance on different hardware (100k rays, shoebox room):

| Hardware | Backend | Time |
|----------|---------|------|
| MacBook Pro M1 | CPU (TBB) | ~2-5s |
| NVIDIA RTX 4090 | CUDA | ~0.3s |
| NVIDIA H100 | CUDA | ~0.1s |

*Mesh mode with BVH adds minimal overhead (<10%) compared to shoebox.*