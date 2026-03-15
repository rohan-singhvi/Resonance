#pragma once

#include "simulation.h"
#include "mesh_loader.h"
#include <vector>

void run_simulation_metal(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir);
