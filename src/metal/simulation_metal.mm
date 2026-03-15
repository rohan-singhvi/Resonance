#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "simulation_metal.h"
#include "simulation.h"
#include "mesh_loader.h"

#include <iostream>
#include <vector>
#include <cstring>

// GPU-side parameter block — POD, layout must match SimParamsGPU in the shader.
// Three packed float[3] fields (12 bytes each) followed by int/float scalars.
struct SimParamsGPU {
    float room_dims[3];
    float source_pos[3];
    float listener_pos[3];
    int   room_type;
    int   num_rays;
    int   ir_length;
    float absorption;
    float scattering;
    float transmission;
    float thickness;
    float listener_radius;
    float sample_rate;
    int   num_triangles;
    int   num_bvh_nodes;
};

// The Metal shader source. Runtime-compiled via newLibraryWithSource:options:error:
// so no xcrun / full Xcode required — Metal.framework alone is sufficient.
static const char* kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// packed_float3 is 12 bytes — matches our C++ float3 struct layout.
struct BVHNodeGPU {
    packed_float3 bb_min;
    packed_float3 bb_max;
    int left_node_index;
    int right_node_index;
    int start;
    int end;
};

struct SimParamsGPU {
    packed_float3 room_dims;
    packed_float3 source_pos;
    packed_float3 listener_pos;
    int   room_type;
    int   num_rays;
    int   ir_length;
    float absorption;
    float scattering;
    float transmission;
    float thickness;
    float listener_radius;
    float sample_rate;
    int   num_triangles;
    int   num_bvh_nodes;
};

// Wang hash — same sequence as the CUDA kernel.
float wang_rand(thread uint& seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return float(seed) / 4294967296.0;
}

// Cosine-weighted hemisphere sample aligned to 'normal' (Malley's method).
float3 cosine_hemisphere(thread uint& seed, float3 normal) {
    float u1 = wang_rand(seed);
    float u2 = wang_rand(seed);
    float r     = sqrt(u1);
    float theta = 6.2831853 * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - u1);
    float3 w = normal;
    float3 a = (fabs(w.x) > 0.9) ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 u_vec = normalize(cross(a, w));
    float3 v_vec = cross(w, u_vec);
    return u_vec * x + v_vec * y + w * z;
}

// Metal has no native float atomicAdd before Metal 3. Use CAS loop.
void atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        desired = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

// Moller-Trumbore triangle intersection. Updates min_dist/out_normal on hit.
void intersect_triangle(
    float3 ro, float3 rd,
    float3 v0, float3 v1, float3 v2, float3 tri_n,
    thread float& min_dist, thread float3& out_normal)
{
    const float eps = 1e-6;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h  = cross(rd, e2);
    float  a  = dot(e1, h);
    if (a > -eps && a < eps) return;
    float  f  = 1.0 / a;
    float3 s  = ro - v0;
    float  u  = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return;
    float3 q  = cross(s, e1);
    float  v  = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) return;
    float  t  = f * dot(e2, q);
    if (t > 1e-3 && t < min_dist) {
        min_dist   = t;
        out_normal = tri_n;
    }
}

kernel void ray_trace_kernel(
    constant SimParamsGPU&    p          [[buffer(0)]],
    device   atomic_uint*     d_ir       [[buffer(1)]],
    constant packed_float3*   d_v0       [[buffer(2)]],
    constant packed_float3*   d_v1       [[buffer(3)]],
    constant packed_float3*   d_v2       [[buffer(4)]],
    constant packed_float3*   d_normals  [[buffer(5)]],
    constant BVHNodeGPU*      d_bvh      [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (int(idx) >= p.num_rays) return;

    uint seed = idx * 1973u + 9277u;

    float3 px = float3(p.source_pos);

    // random initial direction
    float u1 = wang_rand(seed) * 2.0 - 1.0;
    float u2 = wang_rand(seed) * 2.0 - 1.0;
    float u3 = wang_rand(seed) * 2.0 - 1.0;
    float3 dx = normalize(float3(u1, u2, u3));

    float dist_traveled = 0.0;
    float energy        = 1.0;

    for (int bounce = 0; bounce < 50; bounce++) {
        float  min_dist = 1e20;
        float3 nx       = float3(0, 0, 0);

        // shoebox
        if (p.room_type == 0) {
            float3 dims = float3(p.room_dims);
            if (dx.x > 0.0) { float d=(dims.x-px.x)/dx.x; if(d<min_dist){min_dist=d; nx=float3(-1,0,0);} }
            else             { float d=(0.0  -px.x)/dx.x;  if(d<min_dist){min_dist=d; nx=float3( 1,0,0);} }
            if (dx.y > 0.0) { float d=(dims.y-px.y)/dx.y; if(d<min_dist){min_dist=d; nx=float3(0,-1,0);} }
            else             { float d=(0.0  -px.y)/dx.y;  if(d<min_dist){min_dist=d; nx=float3(0, 1,0);} }
            if (dx.z > 0.0) { float d=(dims.z-px.z)/dx.z; if(d<min_dist){min_dist=d; nx=float3(0,0,-1);} }
            else             { float d=(0.0  -px.z)/dx.z;  if(d<min_dist){min_dist=d; nx=float3(0,0, 1);} }
        }
        // dome
        else if (p.room_type == 1) {
            float radius = p.room_dims[0];
            if (dx.y < 0.0) {
                float d = (0.0 - px.y) / dx.y;
                if (d < min_dist) { min_dist = d; nx = float3(0,1,0); }
            }
            float b    = 2.0 * dot(px, dx);
            float c    = dot(px, px) - radius * radius;
            float disc = b*b - 4.0*c;
            if (disc >= 0.0) {
                float sq = sqrt(disc);
                float t1 = (-b - sq) / 2.0;
                float t2 = (-b + sq) / 2.0;
                float t  = (t1 > 1e-3) ? t1 : ((t2 > 1e-3) ? t2 : 1e20);
                if (t < min_dist) { min_dist = t; nx = normalize(px + dx * t); }
            }
        }
        // mesh (iterative BVH traversal)
        else if (p.room_type == 2) {
            float3 inv_dir = float3(1.0/dx.x, 1.0/dx.y, 1.0/dx.z);
            int stack[32];
            int stack_ptr = 0;
            stack[stack_ptr++] = 0;
            while (stack_ptr > 0) {
                int node_idx = stack[--stack_ptr];
                BVHNodeGPU node = d_bvh[node_idx];
                float3 bb_min = float3(node.bb_min);
                float3 bb_max = float3(node.bb_max);
                float3 t1 = (bb_min - px) * inv_dir;
                float3 t2 = (bb_max - px) * inv_dir;
                float tmin = max(max(min(t1.x,t2.x), min(t1.y,t2.y)), min(t1.z,t2.z));
                float tmax = min(min(max(t1.x,t2.x), max(t1.y,t2.y)), max(t1.z,t2.z));
                if (tmax < 0.0 || tmin > tmax || tmin > min_dist) continue;
                if (node.left_node_index == -1) {
                    for (int i = node.start; i < node.end; i++) {
                        intersect_triangle(px, dx,
                            float3(d_v0[i]), float3(d_v1[i]), float3(d_v2[i]),
                            float3(d_normals[i]),
                            min_dist, nx);
                    }
                } else {
                    if (stack_ptr < 30) stack[stack_ptr++] = node.left_node_index;
                    if (stack_ptr < 30) stack[stack_ptr++] = node.right_node_index;
                }
            }
        }

        if (min_dist >= 1e19) break;

        // listener sphere check
        float3 to_l   = float3(p.listener_pos) - px;
        float  t_proj = dot(to_l, dx);
        if (t_proj > 0.0 && t_proj < min_dist) {
            float3 closest  = px + dx * t_proj;
            float3 diff_vec = float3(p.listener_pos) - closest;
            float  dist_sq  = dot(diff_vec, diff_vec);
            float  lr       = p.listener_radius;
            if (dist_sq < lr * lr) {
                float total_dist = dist_traveled + t_proj;
                int   sample_idx = int((total_dist / 343.0) * p.sample_rate);
                if (sample_idx < p.ir_length) {
                    atomic_add_float(&d_ir[sample_idx], energy);
                }
            }
        }

        float rng = wang_rand(seed);
        if (rng < p.transmission) {
            px            = px + dx * (min_dist + p.thickness);
            dist_traveled += min_dist + p.thickness;
            energy        *= p.transmission;
        } else {
            float3 hit    = px + dx * min_dist;
            dist_traveled += min_dist;
            float  d_dot_n = dot(dx, nx);
            float3 spec    = dx - 2.0 * d_dot_n * nx;
            float3 diff    = cosine_hemisphere(seed, nx);
            float  s       = p.scattering;
            dx     = normalize(spec * (1.0 - s) + diff * s);
            px     = hit + nx * 0.001;
            energy *= (1.0 - p.absorption);
        }

        if (energy < 0.001) break;
    }
}
)METAL";


// Build the MTLLibrary once, warn and return nullptr on failure.
static id<MTLLibrary> compile_library(id<MTLDevice> device, NSError** err) {
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion2_4;
    return [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                                options:opts
                                  error:err];
}

void run_simulation_metal(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "[Metal] No Metal device found. Falling back to CPU.\n";
        run_simulation_cpu(params, mesh, ir);
        return;
    }

    std::cout << "Backend: Metal GPU (" << [device.name UTF8String] << ")\n";

    NSError* err = nil;
    id<MTLLibrary> library = compile_library(device, &err);
    if (!library) {
        std::cerr << "[Metal] Shader compile failed: "
                  << [[err localizedDescription] UTF8String] << "\n";
        run_simulation_cpu(params, mesh, ir);
        return;
    }

    id<MTLFunction>            fn  = [library newFunctionWithName:@"ray_trace_kernel"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        std::cerr << "[Metal] PSO creation failed: "
                  << [[err localizedDescription] UTF8String] << "\n";
        run_simulation_cpu(params, mesh, ir);
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    const int ir_len = (int)(params.sample_rate * params.ir_duration_ms / 1000.0f);
    const int N      = params.num_rays;

    // IR buffer — atomic_uint on GPU, read back as float.
    id<MTLBuffer> buf_ir = [device newBufferWithLength:ir_len * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    memset(buf_ir.contents, 0, ir_len * sizeof(uint32_t));

    // Fill GPU parameter block.
    SimParamsGPU gp{};
    gp.room_dims[0]   = params.room_dims.x;
    gp.room_dims[1]   = params.room_dims.y;
    gp.room_dims[2]   = params.room_dims.z;
    gp.source_pos[0]  = params.source_pos.x;
    gp.source_pos[1]  = params.source_pos.y;
    gp.source_pos[2]  = params.source_pos.z;
    gp.listener_pos[0]= params.listener_pos.x;
    gp.listener_pos[1]= params.listener_pos.y;
    gp.listener_pos[2]= params.listener_pos.z;
    gp.room_type      = (int)params.room_type;
    gp.num_rays       = N;
    gp.ir_length      = ir_len;
    float abs_sum = 0.0f;
    for (int b = 0; b < NUM_BANDS; ++b) abs_sum += params.material.absorption[b];
    gp.absorption     = abs_sum / NUM_BANDS;
    gp.scattering     = params.material.scattering;
    gp.transmission   = params.material.transmission;
    gp.thickness      = params.material.thickness;
    gp.listener_radius= params.listener_radius;
    gp.sample_rate    = (float)params.sample_rate;
    gp.num_triangles  = mesh.num_triangles;
    gp.num_bvh_nodes  = (int)mesh.bvh_nodes.size();

    id<MTLBuffer> buf_params = [device newBufferWithBytes:&gp
                                                   length:sizeof(SimParamsGPU)
                                                  options:MTLResourceStorageModeShared];

    // Mesh geometry buffers. Empty 1-element buffers when no mesh (avoids nil binding).
    float3 dummy = make_float3(0, 0, 0);
    auto mesh_buf = [&](const std::vector<float3>& v) -> id<MTLBuffer> {
        if (v.empty()) {
            return [device newBufferWithBytes:&dummy length:sizeof(float3)
                                      options:MTLResourceStorageModeShared];
        }
        return [device newBufferWithBytes:v.data()
                                   length:v.size() * sizeof(float3)
                                  options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> buf_v0      = mesh_buf(mesh.v0);
    id<MTLBuffer> buf_v1      = mesh_buf(mesh.v1);
    id<MTLBuffer> buf_v2      = mesh_buf(mesh.v2);
    id<MTLBuffer> buf_normals = mesh_buf(mesh.normals);

    // BVH nodes — layout matches BVHNodeGPU in shader (verified: both 40 bytes).
    id<MTLBuffer> buf_bvh;
    if (mesh.bvh_nodes.empty()) {
        struct BVHNodeGPUFlat { float d[6]; int i[4]; };
        BVHNodeGPUFlat dummy_node{};
        buf_bvh = [device newBufferWithBytes:&dummy_node length:sizeof(BVHNodeGPUFlat)
                                     options:MTLResourceStorageModeShared];
    } else {
        buf_bvh = [device newBufferWithBytes:mesh.bvh_nodes.data()
                                      length:mesh.bvh_nodes.size() * sizeof(BVHNode)
                                     options:MTLResourceStorageModeShared];
    }

    // Dispatch
    id<MTLCommandBuffer>         cmd     = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:buf_params offset:0 atIndex:0];
    [encoder setBuffer:buf_ir      offset:0 atIndex:1];
    [encoder setBuffer:buf_v0      offset:0 atIndex:2];
    [encoder setBuffer:buf_v1      offset:0 atIndex:3];
    [encoder setBuffer:buf_v2      offset:0 atIndex:4];
    [encoder setBuffer:buf_normals offset:0 atIndex:5];
    [encoder setBuffer:buf_bvh     offset:0 atIndex:6];

    NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
    MTLSize gridSize     = MTLSizeMake((NSUInteger)N, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(tpg, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error) {
        std::cerr << "[Metal] Command buffer error: "
                  << [[cmd.error localizedDescription] UTF8String] << "\n";
    }

    // Read back: the IR was accumulated as uint bit-patterns of floats.
    ir.resize(ir_len, 0.0f);
    const uint32_t* raw = reinterpret_cast<const uint32_t*>(buf_ir.contents);
    for (int i = 0; i < ir_len; i++) {
        memcpy(&ir[i], &raw[i], sizeof(float));
    }

    std::cout << "[Metal] Simulation complete.\n";
}
