#include "simulation.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <atomic> 
#include <mutex>
#include <random>

// TBB Includes
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h> // Required for max_concurrency() check

inline float3 random_cosine_hemisphere(float u1, float u2, float3 normal) {
    float r = sqrtf(u1);
    float theta = 6.2831853f * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1); // Cosine weighted

    // Orthonormal basis construction to align z with 'normal'
    float3 w = normal;
    float3 a = (fabs(w.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    float3 u = normalize(cross(a, w));
    float3 v = cross(w, u);

    return u * x + v * y + w * z;
}

// Ray-Triangle Intersection (Möller–Trumbore) 
// Returns distance t, or 1e20 if miss. Updates normal if hit.
inline float intersect_triangle_cpu(
    const float3& ray_origin, const float3& ray_dir,
    const float3& v0, const float3& v1, const float3& v2,
    const float3& tri_normal,
    float3& out_normal
) {
    const float epsilon = 1e-6f;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    // h = ray_dir X e2
    float3 h;
    h.x = ray_dir.y * e2.z - ray_dir.z * e2.y;
    h.y = ray_dir.z * e2.x - ray_dir.x * e2.z;
    h.z = ray_dir.x * e2.y - ray_dir.y * e2.x;

    float a = dot(e1, h);
    // intentionally allow negative 'a' (backfaces) to ensure we hit 
    // the mesh even if we are inside it or normals are inverted.
    if (a > -epsilon && a < epsilon) return 1e20f; // Parallel

    float f = 1.0f / a;
    float3 s = ray_origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return 1e20f;

    // q = s X e1
    float3 q;
    q.x = s.y * e1.z - s.z * e1.y;
    q.y = s.z * e1.x - s.x * e1.z;
    q.z = s.x * e1.y - s.y * e1.x;

    float v = f * dot(ray_dir, q);
    if (v < 0.0f || u + v > 1.0f) return 1e20f;

    float t = f * dot(e2, q);

    if (t > 1e-3f) {
        out_normal = tri_normal;
        return t;
    }
    return 1e20f;
}


// Recursive BVH Traversal
void traverse_bvh(
    int node_idx, 
    const MeshData& mesh, 
    const float3& origin, const float3& dir, 
    float& closest_dist, float3& best_normal, int& best_tri_idx
) {
    const BVHNode& node = mesh.bvh_nodes[node_idx];

    // box intersection
    float t_box;
    if (!node.bbox.intersect(origin, dir, t_box)) {
        return; // Missed the box, skip entirely!
    }

    // if the box is further away than our current closest hit,
    // there is no point looking inside it.
    if (t_box > closest_dist) return;

    // we check leaf node: checking triangles
    if (node.is_leaf()) {
        float3 temp_n;
        for (int i = node.start; i < node.end; ++i) {
            float dist = intersect_triangle_cpu(
                origin, dir, 
                mesh.v0[i], mesh.v1[i], mesh.v2[i], 
                mesh.normals[i], 
                temp_n
            );

            if (dist < closest_dist) {
                closest_dist = dist;
                best_normal = temp_n;
                best_tri_idx = i;
            }
        }
    } 
    // recurse into children if necessary
    else {
        // We visit children. (Advanced: You could sort them front-to-back 
        // based on t_box distance to terminate earlier, but let's keep it simple first)
        traverse_bvh(node.left_node_index, mesh, origin, dir, closest_dist, best_normal, best_tri_idx);
        traverse_bvh(node.right_node_index, mesh, origin, dir, closest_dist, best_normal, best_tri_idx);
    }
}

void run_simulation_cpu(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir) {
    std::vector<RayPath> debug_paths;
    std::mutex debug_mutex; // because CPU is multi-threaded (TBB)
    int N = params.num_rays;
    int ir_len = (int)(params.sample_rate * params.ir_duration_ms / 1000.0f);
    ir.resize(ir_len, 0.0f);

    const float SPEED_OF_SOUND = 343.0f;
    const float SAMPLE_RATE = (float)params.sample_rate;
    const float LISTENER_RADIUS = params.listener_radius;

    std::cout << "Running CPU Simulation (TBB Threads: " << tbb::this_task_arena::max_concurrency() << ")..." << std::endl;
    
    if (params.room_type == MESH) {
        std::cout << "Mesh Mode, Scene Geometry: " << mesh.num_triangles << " (Accelerated by BVH)" << std::endl;
    }

    // TBB Thread Local Storage
    // It automatically creates a local std::vector<float> for every thread that needs one.
    tbb::enumerable_thread_specific<std::vector<float>> tls_irs([ir_len]() {
        return std::vector<float>(ir_len, 0.0f); // Initializer
    });
    
    // Global Hit Counter
    std::atomic<int> total_hits{0};

    tbb::parallel_for(tbb::blocked_range<int>(0, N), 
        [&](const tbb::blocked_range<int>& range) {

        // Get reference to this thread's local IR buffer
        auto& local_ir = tls_irs.local();

        // Iterate over the chunk assigned to this thread
        for (int i = range.begin(); i != range.end(); ++i) {
        
            // Simple Pseudo-Random Number Generator (PRNG) per ray
            unsigned int seed = i * 747796405 + 2891336453;
            auto rand_float = [&seed]() { 
                seed = seed * 1103515245 + 12345;
                return (float)((seed / 65536) % 32768) / 32768.0f; 
            };

            // random direction vector
            float u = rand_float() * 2.0f - 1.0f;
            float v = rand_float() * 2.0f - 1.0f;
            float w = rand_float() * 2.0f - 1.0f;
            float3 dx = normalize(make_float3(u, v, w));
            float3 px = params.source_pos;

            float dist_traveled = 0.0f;
            float energy = 1.0f;

            // Only record the first 100 rays to save memory/sanity
            bool record_debug = params.debug_rays && (i < 100); 
            RayPath current_path;

            if (record_debug) current_path.push_back(px);

            for (int bounce = 0; bounce < 50; ++bounce) {
                float min_dist = 1e20f;
                float3 nx = make_float3(0,0,0);
                int hit_tri_idx = -1; // debug

                //  shoebox (put this in a function)
                if (params.room_type == SHOEBOX) {
                    if (dx.x > 0.0f) {
                        float d = (params.room_dims.x - px.x) / dx.x;
                        if (d < min_dist) { min_dist = d; nx = make_float3(-1, 0, 0); }
                    } else {
                        float d = (0.0f - px.x) / dx.x;
                        if (d < min_dist) { min_dist = d; nx = make_float3(1, 0, 0); }
                    }
                    if (dx.y > 0.0f) {
                        float d = (params.room_dims.y - px.y) / dx.y;
                        if (d < min_dist) { min_dist = d; nx = make_float3(0, -1, 0); }
                    } else {
                        float d = (0.0f - px.y) / dx.y;
                        if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
                    }
                    if (dx.z > 0.0f) {
                        float d = (params.room_dims.z - px.z) / dx.z;
                        if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, -1); }
                    } else {
                        float d = (0.0f - px.z) / dx.z;
                        if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, 1); }
                    }
                }
                //  dome (put this in a function)
                else if (params.room_type == DOME) {
                    float radius = params.room_dims.x;
                    // Floor
                    if (dx.y < 0.0f) {
                        float d = (0.0f - px.y) / dx.y;
                        if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
                    }
                    // Sphere Intersect 
                    float b = 2.0f * dot(px, dx);
                    float c = dot(px, px) - radius * radius;
                    float disc = b*b - 4.0f*c;
                    if(disc >= 0.0f) {
                        float sqrt_disc = sqrtf(disc);
                        float t1 = (-b - sqrt_disc)/2.0f;
                        float t2 = (-b + sqrt_disc)/2.0f;
                        float t = (t1 > 1e-3f) ? t1 : ((t2 > 1e-3f) ? t2 : 1e20f);
                        if(t < min_dist) {
                            min_dist = t;
                            nx = normalize(px + dx * t);
                        }
                    }
                }
                //  mesh (put this in a function) 
                else if (params.room_type == MESH) {
                    if (!mesh.bvh_nodes.empty()) {
                        traverse_bvh(0, mesh, px, dx, min_dist, nx, hit_tri_idx);
                    } 
                    // Fallback if BVH wasn't built (safety)
                    else {
                        float3 temp_n;
                        for(int t=0; t<mesh.num_triangles; ++t) {
                            float dist = intersect_triangle_cpu(
                                px, dx, 
                                mesh.v0[t], mesh.v1[t], mesh.v2[t], 
                                mesh.normals[t], 
                                temp_n
                            );
                            if (dist < min_dist) {
                                min_dist = dist;
                                nx = temp_n;
                                hit_tri_idx = t;
                            }
                        }
                    }
                }

                //  listener hit?
                float3 to_l = params.listener_pos - px;
                float t_proj = dot(to_l, dx);

                
                // if listener is in front of us and closer than the wall
                if (t_proj > 0 && t_proj < min_dist) {
                    float3 closest = px + dx * t_proj;
                    float dist_sq = length_sq(params.listener_pos - closest);
                    if (dist_sq < LISTENER_RADIUS*LISTENER_RADIUS) {
                        float total_dist = dist_traveled + t_proj;
                        int idx = (int)((total_dist / SPEED_OF_SOUND) * SAMPLE_RATE);
                        if (idx < ir_len) {
                            local_ir[idx] += energy;
                            total_hits++; 
                        }
                    }
                }
                if (min_dist >= 1e19f) {
                    // if (i == 0 && bounce == 0) printf("[Ray 0] Missed Mesh completely.\n");
                    break;
                }

                if (record_debug) {
                    float3 hit_point = px + dx * min_dist;
                    current_path.push_back(hit_point);
                }

                float rng_trans = rand_float();
                
                // if random roll < transmission coeff, we go THROUGH the wall
                if (rng_trans < params.material.transmission) {
                    // move ray through wall (thickness)
                    px = px + dx * (min_dist + params.material.thickness);
                    dist_traveled += min_dist + params.material.thickness;
                    
                    // transmission loss
                    // if transmission is 0.1, we keep 0.1 energy? 
                    // i think we multiply by transmission coeff itself
                    energy *= params.material.transmission; 
                    
                    // Direction does NOT change (refraction ignored for acoustic approximations)
                } 
                else {
                    // reflection (specular + scattering)
                    float3 hit_point = px + dx * min_dist;
                    dist_traveled += min_dist;

                    // Specular Reflection Vector
                    float d_dot_n = dot(dx, nx);
                    float3 spec_dir = dx - 2.0f * d_dot_n * nx;
                    
                    // Diffuse Reflection Vector (Lambertian)
                    // We need two random numbers for the hemisphere sampling
                    float r1 = rand_float();
                    float r2 = rand_float();
                    float3 diff_dir = random_cosine_hemisphere(r1, r2, nx);

                    // Mix based on Scattering Coefficient
                    float s = params.material.scattering;
                    float3 mixed_dir = spec_dir * (1.0f - s) + diff_dir * s;
                    dx = normalize(mixed_dir);

                    // Nudge off wall to prevent self-intersection
                    px = hit_point + nx * 0.001f;

                    // Absorption
                    energy *= (1.0f - params.material.absorption);
                }

                if (record_debug) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    debug_paths.push_back(current_path);
                }

                if (energy < 0.001f) break;
            }
        }
    });

    std::cout << "Simulation Complete. Total Listener Hits: " << total_hits << std::endl;
    if (total_hits == 0) {
        std::cout << "WARNING: No rays hit the listener! Try increasing --rays or the listener size is too small." << std::endl;
    }

    // We iterate through the enumerable_thread_specific storage and sum them up
    for(const auto& local_buffer : tls_irs) {
        for(int i=0; i<ir_len; ++i) {
            ir[i] += local_buffer[i];
        }
    }
    if (params.debug_rays)
    {
        std::cout << "Saving debug rays..." << std::endl;
        save_rays_to_obj("debug_rays.obj", debug_paths);
    }
}