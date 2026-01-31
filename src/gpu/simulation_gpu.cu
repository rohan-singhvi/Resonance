#include "simulation.cuh"
#include "cuda_math.h"
#include <cstdio>

#define SPEED_OF_SOUND 343.0f
#define SAMPLE_RATE 44100.0f
#define LISTENER_RADIUS 0.5f
#define MAX_BOUNCES 50

__device__ bool intersect_aabb(
    const float3& ray_origin, const float3& ray_inv_dir, 
    const float3& box_min, const float3& box_max, 
    float& t_min_out
) {
    float tmin = -1e20f;
    float tmax = 1e20f;

    // X-Axis
    {
        float t1 = (box_min.x - ray_origin.x) * ray_inv_dir.x;
        float t2 = (box_max.x - ray_origin.x) * ray_inv_dir.x;
        tmin = fmaxf(tmin, fminf(t1, t2));
        tmax = fminf(tmax, fmaxf(t1, t2));
    }

    // Y-Axis
    {
        float t1 = (box_min.y - ray_origin.y) * ray_inv_dir.y;
        float t2 = (box_max.y - ray_origin.y) * ray_inv_dir.y;
        tmin = fmaxf(tmin, fminf(t1, t2));
        tmax = fminf(tmax, fmaxf(t1, t2));
    }

    // Z-Axis
    {
        float t1 = (box_min.z - ray_origin.z) * ray_inv_dir.z;
        float t2 = (box_max.z - ray_origin.z) * ray_inv_dir.z;
        tmin = fmaxf(tmin, fminf(t1, t2));
        tmax = fminf(tmax, fmaxf(t1, t2));
    }

    if (tmax >= tmin && tmax > 0.0f) {
        t_min_out = tmin;
        return true;
    }
    return false;
}

// rand num gen (wang hash)
// statistically decent for ray tracing visuals/acoustics
__device__ float rand_gpu(unsigned int& seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)(seed) / 4294967296.0f;
}

// random hemisphere vector
__device__ float3 random_hemisphere_gpu(unsigned int& seed, float3 normal) {
    float u1 = rand_gpu(seed);
    float u2 = rand_gpu(seed);
    
    float r = sqrtf(u1);
    float theta = 6.2831853f * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1);

    float3 w = normal;
    // Handle the case where w is parallel to (1,0,0)
    float3 a = (fabs(w.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    float3 u = normalize(cross(a, w));
    float3 v = cross(w, u);
    
    return u * x + v * y + w * z;
}

__device__ void intersect_sphere(
    const float3& ray_origin, const float3& ray_dir, 
    float radius, 
    float& min_dist, float3& normal
) {
    // Ray-Sphere intersection
    float b = 2.0f * dot(ray_origin, ray_dir);
    float c = dot(ray_origin, ray_origin) - (radius * radius);
    float disc = b * b - 4.0f * c;

    if (disc < 0.0f) return;

    float sqrt_disc = sqrtf(disc);
    float t1 = (-b - sqrt_disc) / 2.0f;
    float t2 = (-b + sqrt_disc) / 2.0f;

    float t = 1e20f;
    if (t1 > 1e-3f) t = t1;
    else if (t2 > 1e-3f) t = t2;

    if (t < min_dist) {
        min_dist = t;
        float3 hit_point = ray_origin + ray_dir * t;
        normal = normalize(hit_point); // Sphere at 0,0,0 normal is just normalized pos
    }
}

__device__ void intersect_triangle(
    const float3& ray_origin, const float3& ray_dir,
    const float3& v0, const float3& v1, const float3& v2,
    const float3& tri_normal,
    float& min_dist, float3& normal
) {
    const float epsilon = 1e-6f;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    float3 h;
    h.x = ray_dir.y * e2.z - ray_dir.z * e2.y;
    h.y = ray_dir.z * e2.x - ray_dir.x * e2.z;
    h.z = ray_dir.x * e2.y - ray_dir.y * e2.x;

    float a = dot(e1, h);
    if (a > -epsilon && a < epsilon) return;

    float f = 1.0f / a;
    float3 s = ray_origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return;

    float3 q;
    q.x = s.y * e1.z - s.z * e1.y;
    q.y = s.z * e1.x - s.x * e1.z;
    q.z = s.x * e1.y - s.y * e1.x;

    float v = f * dot(ray_dir, q);
    if (v < 0.0f || u + v > 1.0f) return;

    float t = f * dot(e2, q);

    if (t > 1e-3f && t < min_dist) {
        min_dist = t;
        normal = tri_normal;
    }
}

__global__ void ray_trace_kernel(
    float3* d_pos,
    float3* d_dir,
    float3 room_dims,
    float3 listener_pos,
    float* d_impulse_response,
    int room_type,
    int ir_length,
    MaterialParams mat,
    // mesh data
    float3* d_v0, float3* d_v1, float3* d_v2, float3* d_normals, int num_triangles,
    BVHNode* d_bvh_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = idx * 1973 + 9277;
    
    // bound check (assuming d_pos size matches num_rays)
    // can't easily check array size in CUDA, rely on launch bounds
    
    float3 px = d_pos[idx];
    float3 dx = d_dir[idx];
    
    float dist_traveled = 0.0f;
    float energy = 1.0f;

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        float min_dist = 1e20f;
        float3 nx = make_float3(0.0f, 0.0f, 0.0f);

        //  shoebox
        if (room_type == SHOEBOX) {
            // X-Walls
            if (dx.x > 0.0f) {
                float d = (room_dims.x - px.x) / dx.x;
                if (d < min_dist) { min_dist = d; nx = make_float3(-1, 0, 0); }
            } else {
                float d = (0.0f - px.x) / dx.x;
                if (d < min_dist) { min_dist = d; nx = make_float3(1, 0, 0); }
            }
            // Y-Walls
            if (dx.y > 0.0f) {
                float d = (room_dims.y - px.y) / dx.y;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, -1, 0); }
            } else {
                float d = (0.0f - px.y) / dx.y;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
            }
            // Z-Walls
            if (dx.z > 0.0f) {
                float d = (room_dims.z - px.z) / dx.z;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, -1); }
            } else {
                float d = (0.0f - px.z) / dx.z;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, 1); }
            }
        }
        //  dome 
        else if (room_type == DOME) {
            float radius = room_dims.x;
            // Floor
            if (dx.y < 0.0f) {
                float d = (0.0f - px.y) / dx.y;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
            }
            // Dome
            intersect_sphere(px, dx, radius, min_dist, nx);
        }
        //  mesh 
        else if (room_type == MESH) {
            float3 inv_dir = make_float3(1.0f/dx.x, 1.0f/dx.y, 1.0f/dx.z);

            // Since we can't recurse, we manage a stack manually.
            // A depth of 32 is enough for 4 billion triangle (2^32).
            int stack[32];
            int stack_ptr = 0;
            
            // Push Root Node (Index 0)
            stack[stack_ptr++] = 0;

            while (stack_ptr > 0) {
                // Pop a node
                int node_idx = stack[--stack_ptr];
                
                // Fetch node data from global memory
                // We read it into a local register to be fast
                BVHNode node = d_bvh_nodes[node_idx];

                // AABB Intersection
                float t_box;
                if (!intersect_aabb(px, inv_dir, node.bbox.min, node.bbox.max, t_box)) {
                    continue; // Missed the box, ignore children
                }

                // If the box is further than our best hit, skip it
                if (t_box > min_dist) continue;

                // leaf?
                // check specific triangles
                if (node.left_node_index == -1) { // is_leaf logic
                    for (int i = node.start; i < node.end; ++i) {
                        float3 v0 = d_v0[i];
                        float3 v1 = d_v1[i];
                        float3 v2 = d_v2[i];
                        float3 tri_norm = d_normals[i];

                        intersect_triangle(px, dx, v0, v1, v2, tri_norm, min_dist, nx);
                    }
                } 
                // add children to stack
                else {
                    stack[stack_ptr++] = node.left_node_index;
                    stack[stack_ptr++] = node.right_node_index;
                }
            }
        }
        
        // Missed everything?
        if (min_dist >= 1e19f) break;

        // listener intersection along this ray segment
        float3 to_listener = listener_pos - px;
        float t_proj = dot(to_listener, dx);

        if (t_proj > 0.0f && t_proj < min_dist) {
            float3 closest_point = px + dx * t_proj;
            float dist_sq = length_sq(listener_pos - closest_point);
            
            if (dist_sq < (LISTENER_RADIUS * LISTENER_RADIUS)) {
                // hit!
                float total_dist = dist_traveled + t_proj;
                int idx_time = (int)((total_dist / SPEED_OF_SOUND) * SAMPLE_RATE);
                
                if (idx_time < ir_length) {
                    atomicAdd(&d_impulse_response[idx_time], energy);
                }
            }
        }

        float rng_val = rand_gpu(seed);

        // Transmission Check
        if (rng_val < mat.transmission) {
            // PASS THROUGH
            px = px + dx * (min_dist + mat.thickness);
            dist_traveled += min_dist + mat.thickness;
            energy *= mat.transmission; 
            // Direction 'dx' stays same
        } 
        else {
            // REFLECTION
            float3 hit_point = px + dx * min_dist;
            dist_traveled += min_dist;

            // Specular
            float d_dot_n = dot(dx, nx);
            float3 spec = dx - 2.0f * d_dot_n * nx;

            // Diffuse
            float3 diffuse = random_hemisphere_gpu(seed, nx);

            // Mix
            float3 mixed = spec * (1.0f - mat.scattering) + diffuse * mat.scattering;
            dx = normalize(mixed);
            
            // Nudge
            px = hit_point + nx * 0.001f;
            
            // Absorption
            energy *= (1.0f - mat.absorption);
        }

        if (energy < 0.001f) break;
    }
}

// wrapper
void run_simulation_gpu(const SimulationParams& params, const MeshData& mesh, std::vector<float>& h_impulse_response) {
    int N = params.num_rays;
    
    // generate rays
    std::vector<float3> h_pos(N);
    std::vector<float3> h_dir(N);
    
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        h_pos[i] = params.source_pos;
        
        // random spherical direction
        float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float3 d = make_float3(u, v, w);
        h_dir[i] = normalize(d);
    }

    // allocate device memory
    float3 *d_pos, *d_dir, *d_v0 = nullptr, *d_v1 = nullptr, *d_v2 = nullptr, *d_normals = nullptr;
    float *d_ir;
    
    cudaMalloc(&d_pos, N * sizeof(float3));
    cudaMalloc(&d_dir, N * sizeof(float3));
    
    cudaMemcpy(d_pos, h_pos.data(), N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dir, h_dir.data(), N * sizeof(float3), cudaMemcpyHostToDevice);

    // IR buffer
    int ir_len = 44100; // 1 second buffer
    cudaMalloc(&d_ir, ir_len * sizeof(float));
    cudaMemset(d_ir, 0, ir_len * sizeof(float));
    
    BVHNode* d_bvh_nodes = nullptr;

    // mesh buffer (if needed)
    if (params.room_type == MESH) {
        int t_count = mesh.num_triangles;
        cudaMalloc(&d_v0, t_count * sizeof(float3));
        cudaMalloc(&d_v1, t_count * sizeof(float3));
        cudaMalloc(&d_v2, t_count * sizeof(float3));
        cudaMalloc(&d_normals, t_count * sizeof(float3));
        
        cudaMemcpy(d_v0, mesh.v0.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1, mesh.v1.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, mesh.v2.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_normals, mesh.normals.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
        if (!mesh.bvh_nodes.empty()) {
            size_t node_size = mesh.bvh_nodes.size() * sizeof(BVHNode);
            cudaMalloc(&d_bvh_nodes, node_size);
            cudaMemcpy(d_bvh_nodes, mesh.bvh_nodes.data(), node_size, cudaMemcpyHostToDevice);
            printf("Copied %lu BVH nodes to GPU\n", mesh.bvh_nodes.size());
        }
    }
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("Launching Kernel: %d rays, %d blocks\n", N, blocks);
    
    ray_trace_kernel<<<blocks, threads>>>(
        d_pos, d_dir, 
        params.room_dims, 
        params.listener_pos, 
        d_ir, 
        (int)params.room_type, 
        ir_len,
        params.material,
        d_v0, d_v1, d_v2, d_normals, mesh.num_triangles,
        d_bvh_nodes
    );
    
    cudaDeviceSynchronize();
    
    // errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    h_impulse_response.resize(ir_len);
    cudaMemcpy(h_impulse_response.data(), d_ir, ir_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_pos);
    cudaFree(d_dir);
    cudaFree(d_ir);
    if(d_v0) cudaFree(d_v0);
    if(d_v1) cudaFree(d_v1);
    if(d_v2) cudaFree(d_v2);
    if(d_normals) cudaFree(d_normals);
    if(d_bvh_nodes) cudaFree(d_bvh_nodes);
}
