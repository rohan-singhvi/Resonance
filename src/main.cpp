#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "simulation.h"
#include "convolution.h"
#include "wav_io.h"
#include "mesh_loader.h"

void print_usage() {
    std::cout << "Usage: ./acoustic_sim [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --room <type>      shoebox (default), dome, mesh\n";
    std::cout << "  --dims <x,y,z>     Room dimensions or Radius\n";
    std::cout << "  --mesh <file>      Path to .obj file\n";
    std::cout << "  --rays <n>         Number of rays (default 100000)\n";
    std::cout << "  --input <file>     Input audio file\n";
    std::cout << "  --mix <0.0-1.0>    Reverb mix (default 0.4)\n";
    std::cout << "  --out <file>       Output file\n";
    std::cout << "  --absorption <0-1> Wall absorption (0=Reflective, 1=Dead)\n";
    std::cout << "  --scattering <0-1> Wall roughness (0=Mirror, 1=Diffuse)\n";
    std::cout << "  --trans <0-1>      Transmission (0=Opaque, 1=Transparent)\n";
    std::cout << "  --thick <meters>   Wall thickness (default 0.1)\n";
    std::cout << "  --debug            Outputs a debug OBJ file to visualize rays\n";
}

float3 parse_dims(const char* arg) {
    float x = 0, y = 0, z = 0;
    sscanf(arg, "%f,%f,%f", &x, &y, &z);
    return make_float3(x, y, z);
}

void run_simulation(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir) {
#ifdef ENABLE_CUDA
    std::cout << "Backend: CUDA GPU" << std::endl;
    run_simulation_gpu(params, mesh, ir);
#else
    std::cout << "Backend: CPU" << std::endl;
    run_simulation_cpu(params, mesh, ir);
#endif
}

std::vector<float> apply_reverb(const std::vector<float>& dry, const std::vector<float>& ir, float mix) {
#ifdef ENABLE_CUDA
    return apply_reverb_gpu(dry, ir, mix);
#else
    return apply_reverb_cpu(dry, ir, mix);
#endif
}

int main(int argc, char** argv) {
    SimulationParams params;
    params.num_rays = 100000;
    params.room_type = SHOEBOX;
    params.room_dims = make_float3(10.0f, 5.0f, 3.0f);
    params.source_pos = make_float3(2.0f, 1.5f, 1.5f);
    params.listener_pos = make_float3(8.0f, 1.5f, 1.5f);
    
    params.material.absorption = 0.10f;  // Concrete (Reflective)
    params.material.scattering = 0.10f;  // Smooth surface
    params.material.transmission = 0.0f; // Solid walls
    params.material.thickness = 0.2f;    // 20cm
    
    std::string outfile = "out.wav";
    std::string input_audio_file = "";
    float mix = 0.4f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--room") == 0 && i + 1 < argc) {
            std::string type = argv[++i];
            if (type == "shoebox") params.room_type = SHOEBOX;
            else if (type == "dome") params.room_type = DOME;
            else if (type == "mesh") params.room_type = MESH;
        }
        else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) params.room_dims = parse_dims(argv[++i]);
        else if (strcmp(argv[i], "--mesh") == 0 && i + 1 < argc) params.mesh_path = argv[++i];
        else if (strcmp(argv[i], "--rays") == 0 && i + 1 < argc) params.num_rays = atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) outfile = argv[++i];
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_audio_file = argv[++i];
        else if (strcmp(argv[i], "--mix") == 0 && i + 1 < argc) mix = atof(argv[++i]);
        
        else if (strcmp(argv[i], "--absorption") == 0 && i + 1 < argc) params.material.absorption = atof(argv[++i]);
        else if (strcmp(argv[i], "--scattering") == 0 && i + 1 < argc) params.material.scattering = atof(argv[++i]);
        else if (strcmp(argv[i], "--trans") == 0 && i + 1 < argc) params.material.transmission = atof(argv[++i]);
        else if (strcmp(argv[i], "--thick") == 0 && i + 1 < argc) params.material.thickness = atof(argv[++i]);
        else if (strcmp(argv[i], "--debug") == 0) params.debug_rays = true;
        else if (strcmp(argv[i], "--help") == 0) { print_usage(); return 0; }
    }

    MeshData mesh;
    if (params.room_type == MESH) {
        if (params.mesh_path.empty()) {
            std::cerr << "Error: --mesh argument required for mesh room type.\n";
            return 1;
        }
        mesh = load_obj(params.mesh_path);
        build_bvh(mesh);
    }

    std::cout << "Starting Simulation (" << params.num_rays << " rays)...\n";
    std::cout << "Mat Props -> Abs: " << params.material.absorption 
              << ", Scat: " << params.material.scattering 
              << ", Trans: " << params.material.transmission << "\n";

    std::vector<float> impulse_response;
    run_simulation(params, mesh, impulse_response);

    if (!input_audio_file.empty()) {
        std::cout << "Loading input audio: " << input_audio_file << "...\n";
        WavData input = read_wav(input_audio_file);
        
        if (!input.success) {
            std::cerr << "Failed to load input audio.\n";
            return 1;
        }

        std::cout << "Applying Reverb (Convolution)... Mix: " << mix << "\n";
        std::vector<float> wet_result = apply_reverb(input.samples, impulse_response, mix);
        write_wav(outfile, wet_result, input.sample_rate);
    } 
    else {
        std::cout << "No input audio provided. Saving raw Room Impulse Response.\n";
        write_wav(outfile, impulse_response, 44100);
    }
    
    return 0;
}