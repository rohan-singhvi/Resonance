#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <sstream>
#include "simulation.h"
#include "convolution.h"
#include "wav_io.h"
#include "mesh_loader.h"
#include "material_presets.h"
#ifdef ENABLE_METAL
    #include "simulation_metal.h"
#endif

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
    std::cout << "  --material <name>         Preset material (concrete, carpet_thick, glass, ...)\n";
    std::cout << "  --absorption <0-1>        Broadband absorption, all bands set uniformly\n";
    std::cout << "  --scattering <0-1>       Wall roughness (0=Mirror, 1=Diffuse)\n";
    std::cout << "  --trans <0-1>            Transmission (0=Opaque, 1=Transparent)\n";
    std::cout << "  --thick <meters>         Wall thickness (default 0.2)\n";
    std::cout << "  --listener-radius <m>    Listener sphere radius (default 0.5)\n";
    std::cout << "  --sr <hz>                Sample rate: 44100, 48000, 96000 (default 44100)\n";
    std::cout << "  --ir-len <ms>            IR duration in milliseconds (default 1000)\n";
    std::cout << "  --air-absorption <c>     Air absorption coefficient per meter (default 0.001)\n";
    std::cout << "  --early-ms <ms>          Early reflection cutoff in ms (default 80)\n";
    std::cout << "  --debug                  Outputs a debug OBJ file to visualize rays\n";
    std::cout << "  --mat-assign <assignments>  Per-group material: \"floor=carpet_thick,walls=concrete\"\n";
}

float3 parse_dims(const char* arg) {
    float x = 0, y = 0, z = 0;
    sscanf(arg, "%f,%f,%f", &x, &y, &z);
    return make_float3(x, y, z);
}

void run_simulation(
    const SimulationParams& params,
    const MeshData& mesh,
    std::vector<float>& ir,
    std::vector<std::vector<float>>& ir_bands,
    std::vector<float>& ir_early,
    std::vector<float>& ir_late
) {
#ifdef ENABLE_CUDA
    std::cout << "Backend: CUDA GPU" << std::endl;
    run_simulation_gpu(params, mesh, ir, ir_bands, ir_early, ir_late);
#elif defined(ENABLE_METAL)
    std::cout << "Backend: Metal GPU" << std::endl;
    run_simulation_metal(params, mesh, ir);
#else
    std::cout << "Backend: CPU" << std::endl;
    run_simulation_cpu(params, mesh, ir, ir_bands, ir_early, ir_late);
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

    params.material.set_uniform_absorption(0.10f);
    params.material.scattering = 0.10f;
    params.material.transmission = 0.0f;
    params.material.thickness = 0.2f;
    params.listener_radius = 0.5f;
    params.sample_rate = 44100;
    params.ir_duration_ms = 1000.0f;
    params.air_absorption = 0.001f;
    params.early_reflection_ms = 80.0f;

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

        else if (strcmp(argv[i], "--absorption") == 0 && i + 1 < argc) params.material.set_uniform_absorption(atof(argv[++i]));
        else if (strcmp(argv[i], "--material") == 0 && i + 1 < argc) {
            if (!MaterialPresets::lookup(argv[++i], params.material.absorption)) {
                std::cerr << "Unknown material: " << argv[i] << "\nAvailable:\n";
                MaterialPresets::list_names();
                return 1;
            }
        }
        else if (strcmp(argv[i], "--scattering") == 0 && i + 1 < argc) params.material.scattering = atof(argv[++i]);
        else if (strcmp(argv[i], "--trans") == 0 && i + 1 < argc) params.material.transmission = atof(argv[++i]);
        else if (strcmp(argv[i], "--thick") == 0 && i + 1 < argc) params.material.thickness = atof(argv[++i]);
        else if (strcmp(argv[i], "--debug") == 0) params.debug_rays = true;
        else if (strcmp(argv[i], "--listener-radius") == 0 && i + 1 < argc) params.listener_radius = atof(argv[++i]);
        else if (strcmp(argv[i], "--sr") == 0 && i + 1 < argc) params.sample_rate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ir-len") == 0 && i + 1 < argc) params.ir_duration_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--air-absorption") == 0 && i + 1 < argc) params.air_absorption = atof(argv[++i]);
        else if (strcmp(argv[i], "--early-ms") == 0 && i + 1 < argc) params.early_reflection_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--mat-assign") == 0 && i + 1 < argc) params.mat_assign = argv[++i];
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

    // Build scene_materials: one entry per OBJ group (or just [global] for non-mesh).
    if (params.room_type == MESH && !mesh.group_names.empty()) {
        params.scene_materials.assign(mesh.group_names.size(), params.material);
    } else {
        params.scene_materials.push_back(params.material);
    }

    // Process --mat-assign "floor=carpet_thick,walls=concrete"
    if (!params.mat_assign.empty() && !mesh.group_names.empty()) {
        std::stringstream ss(params.mat_assign);
        std::string token;
        while (std::getline(ss, token, ',')) {
            size_t eq = token.find('=');
            if (eq == std::string::npos) continue;
            std::string group  = token.substr(0, eq);
            std::string preset = token.substr(eq + 1);
            auto it = std::find(mesh.group_names.begin(), mesh.group_names.end(), group);
            if (it == mesh.group_names.end()) {
                std::cerr << "Warning: group '" << group << "' not found in mesh\n";
                continue;
            }
            int idx = (int)(it - mesh.group_names.begin());
            if (!MaterialPresets::lookup(preset.c_str(), params.scene_materials[idx].absorption)) {
                std::cerr << "Warning: unknown material preset '" << preset << "'\n";
            }
        }
    }

    // Print group -> material mapping for mesh rooms.
    if (params.room_type == MESH && !mesh.group_names.empty()) {
        std::cout << "Surface material assignments:\n";
        for (int g = 0; g < (int)mesh.group_names.size(); ++g) {
            float mean_abs = 0.0f;
            for (int b = 0; b < NUM_BANDS; ++b)
                mean_abs += params.scene_materials[g].absorption[b];
            mean_abs /= NUM_BANDS;
            std::cout << "  group[" << g << "] '" << mesh.group_names[g]
                      << "' -> mean absorption " << mean_abs << "\n";
        }
    }

    std::cout << "Starting Simulation (" << params.num_rays << " rays)...\n";
    std::cout << "Mat Props -> Scat: " << params.material.scattering
              << ", Trans: " << params.material.transmission << "\n";

    std::vector<float> impulse_response;
    std::vector<std::vector<float>> ir_bands;
    std::vector<float> ir_early;
    std::vector<float> ir_late;
    run_simulation(params, mesh, impulse_response, ir_bands, ir_early, ir_late);

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
        write_wav(outfile, impulse_response, params.sample_rate);
        write_wav("out_early.wav", ir_early, params.sample_rate);
        write_wav("out_late.wav", ir_late, params.sample_rate);

        static const char* band_names[] = {"125hz","250hz","500hz","1khz","2khz","4khz","8khz"};
        for (int b = 0; b < NUM_BANDS; ++b) {
            std::string bname = std::string("out_band_") + band_names[b] + ".wav";
            write_wav(bname, ir_bands[b], params.sample_rate);
        }
    }

    return 0;
}
