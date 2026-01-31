#include <fstream>
#include <vector>
#include <string>
#include <iostream>

#include "cuda_math.h" // for float3

// Each path is just a list of points (Origin -> Hit -> Hit -> ...)
using RayPath = std::vector<float3>;

inline void save_rays_to_obj(const std::string& filename, const std::vector<RayPath>& rays) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing debug rays.\n";
        return;
    }

    out << "# Ray Visualization\n";
    
    int global_vertex_offset = 1; // OBJ indices start at 1

    for (const auto& path : rays) {
        if (path.size() < 2) continue; // Need at least 2 points to make a line

        // Write Vertices for this specific ray
        for (const auto& p : path) {
            out << "v " << p.x << " " << p.y << " " << p.z << "\n";
        }

        // Write Line Indices (connect the dots)
        // If a ray has 3 points (A, B, C), we write: l 1 2 3
        out << "l";
        for (int i = 0; i < path.size(); ++i) {
            out << " " << (global_vertex_offset + i);
        }
        out << "\n";

        global_vertex_offset += path.size();
    }
    
    std::cout << "Saved " << rays.size() << " rays to " << filename << "\n";
}