#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include <vector>
#include <cuda_runtime.h>
#include "cuda_math.h"
#include "mesh_loader.h"

static constexpr int NUM_BANDS = 7;

struct MaterialParams {
    float absorption[NUM_BANDS];
    float scattering;
    float transmission;
    float thickness;
};

enum RoomType {
    SHOEBOX = 0,
    DOME = 1,
    MESH = 2
};

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

// Host wrapper to launch kernel
void run_acoustic_simulation(
    const SimulationParams& params, 
    const MeshData& mesh, 
    std::vector<float>& h_impulse_response
);

#endif