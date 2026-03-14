#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <string>
#include "debug.h"
#include "cuda_math.h"
#include "mesh_loader.h"

static constexpr int NUM_BANDS = 7;

struct MaterialParams {
    float absorption[NUM_BANDS]; // per-octave-band: 125, 250, 500, 1k, 2k, 4k, 8k Hz
    float scattering;
    float transmission;
    float thickness;

    void set_uniform_absorption(float v) {
        for (int i = 0; i < NUM_BANDS; ++i) absorption[i] = v;
    }
};

enum RoomType { SHOEBOX = 0, DOME = 1, MESH = 2 };

struct SimulationParams {
    int num_rays;
    RoomType room_type;
    float3 room_dims; // Box: L,W,H | Dome: R,0,0
    float3 source_pos;
    float3 listener_pos;
    std::string mesh_path;
    MaterialParams material;
    bool debug_rays = false;
    float listener_radius = 0.5f;
    int sample_rate = 44100;
    float ir_duration_ms = 1000.0f;
    float air_absorption = 0.001f;
    float early_reflection_ms = 80.0f;
};

// Main Entry Point (Dispatcher)
void run_simulation(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir);

void run_simulation_cpu(
    const SimulationParams& params,
    const MeshData& mesh,
    std::vector<float>& ir,
    std::vector<std::vector<float>>& ir_bands,
    std::vector<float>& ir_early,
    std::vector<float>& ir_late
);

#ifdef ENABLE_CUDA
void run_simulation_gpu(
    const SimulationParams& params,
    const MeshData& mesh,
    std::vector<float>& ir,
    std::vector<std::vector<float>>& ir_bands,
    std::vector<float>& ir_early,
    std::vector<float>& ir_late
);
#endif

#endif