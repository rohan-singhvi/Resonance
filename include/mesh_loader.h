#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "cuda_math.h"

// for optimization we will need bounding volumes
// this is an axis-aligned bounding box
struct AABB {
    float3 min;
    float3 max;
    // ctor - this is initially an invalid box
    // which will be fixed when we fit points into it
    AABB() {
        float inf = std::numeric_limits<float>::infinity();
        min = make_float3(inf, inf, inf);
        max = make_float3(-inf, -inf, -inf);
    }

    void fit(const float3& point) {
        min.x = std::fmin(min.x, point.x);
        min.y = std::fmin(min.y, point.y);
        min.z = std::fmin(min.z, point.z);
        max.x = std::fmax(max.x, point.x);
        max.y = std::fmax(max.y, point.y);
        max.z = std::fmax(max.z, point.z);
    }
    // Returns true if ray (origin + t*dir) hits the box.
    // t_hit is the distance to the box (helpful for sorting).
    inline bool intersect(const float3& origin, const float3& dir, float& t_hit) const {
        // We start with an infinite time interval [t_min, t_max]
        float t_min = 0.0f;
        float t_max = 1e20f;

        // Check X slab
        if (fabs(dir.x) < 1e-6f) {
            // Ray is parallel to X plane. If origin is outside, we miss.
            if (origin.x < min.x || origin.x > max.x) return false;
        } else {
            float inv_d = 1.0f / dir.x;
            float t1 = (min.x - origin.x) * inv_d;
            float t2 = (max.x - origin.x) * inv_d;
            // Narrow the interval
            t_min = std::fmax(t_min, std::fmin(t1, t2));
            t_max = std::fmin(t_max, std::fmax(t1, t2));
        }

        // Check Y slab
        if (fabs(dir.y) < 1e-6f) {
            if (origin.y < min.y || origin.y > max.y) return false;
        } else {
            float inv_d = 1.0f / dir.y;
            float t1 = (min.y - origin.y) * inv_d;
            float t2 = (max.y - origin.y) * inv_d;
            t_min = std::fmax(t_min, std::fmin(t1, t2));
            t_max = std::fmin(t_max, std::fmax(t1, t2));
        }

        // Check Z slab
        if (fabs(dir.z) < 1e-6f) {
            if (origin.z < min.z || origin.z > max.z) return false;
        } else {
            float inv_d = 1.0f / dir.z;
            float t1 = (min.z - origin.z) * inv_d;
            float t2 = (max.z - origin.z) * inv_d;
            t_min = std::fmax(t_min, std::fmin(t1, t2));
            t_max = std::fmin(t_max, std::fmax(t1, t2));
        }

        // Did we survive?
        if (t_max >= t_min) {
            t_hit = t_min;
            return true;
        }
        return false;
    }
};

// Building Volume Hierarchies node
struct BVHNode {
    AABB bbox;
    int left_node_index = -1;   // index of left child node
    int right_node_index = -1;  // index of right child node
    int start = 0;  // start index of triangles in this node
    int end = 0;    // end index of triangles in this node
    int triangle_count() const { return end - start; }
    bool is_leaf() const { return left_node_index == -1 && right_node_index == -1; }
};


struct MeshData {
    std::vector<float3> v0;
    std::vector<float3> v1;
    std::vector<float3> v2;
    std::vector<float3> normals;
    int num_triangles = 0;
    std::vector<BVHNode> bvh_nodes;
    std::vector<int> material_ids;        // one per triangle, indexes into scene_materials
    std::vector<std::string> group_names; // unique group names from OBJ 'g' directives
};


// centroid of triangle tri_index
inline float3 get_centroid(const MeshData& mesh, int tri_index) {
    float3 p0 = mesh.v0[tri_index];
    float3 p1 = mesh.v1[tri_index];
    float3 p2 = mesh.v2[tri_index];
    return make_float3((p0.x + p1.x + p2.x) / 3.0f,
                       (p0.y + p1.y + p2.y) / 3.0f,
                       (p0.z + p1.z + p2.z) / 3.0f);
}

// update the bounding box of a node by iterating all its triangles
inline void update_node_bounds(int node_idx, MeshData& mesh){
    BVHNode& node = mesh.bvh_nodes[node_idx];
    AABB bbox;
    for(int i = node.start; i < node.end; i++){
        bbox.fit(mesh.v0[i]);
        bbox.fit(mesh.v1[i]);
        bbox.fit(mesh.v2[i]);
    }
    node.bbox = bbox;
}

// SAH BVH helpers

inline float aabb_surface_area(const AABB& b) {
    float3 d = b.max - b.min;
    return 2.0f * (d.x*d.y + d.y*d.z + d.z*d.x);
}

inline AABB merge_aabb(const AABB& a, const AABB& b) {
    AABB r;
    r.min = make_float3(std::fmin(a.min.x, b.min.x), std::fmin(a.min.y, b.min.y), std::fmin(a.min.z, b.min.z));
    r.max = make_float3(std::fmax(a.max.x, b.max.x), std::fmax(a.max.y, b.max.y), std::fmax(a.max.z, b.max.z));
    return r;
}

// Binned SAH subdivide — replaces the old median-split.
// Uses 16 bins per axis, evaluates all 3 axes, picks the split
// with the lowest SAH cost. Falls back to leaf when no beneficial
// split exists or the node already has <= 2 triangles.
inline void subdivide(int node_idx, MeshData& mesh) {
    // Read start/end before any push_back that would invalidate the reference.
    int start = mesh.bvh_nodes[node_idx].start;
    int end   = mesh.bvh_nodes[node_idx].end;
    int count = end - start;

    if (count <= 2) return;

    const int NUM_BINS = 16;
    float best_cost  = (float)count; // leaf cost
    int   best_axis  = -1;
    float best_split = 0.0f;

    float parent_sa = aabb_surface_area(mesh.bvh_nodes[node_idx].bbox);
    if (parent_sa < 1e-10f) return; // degenerate flat geometry

    for (int axis = 0; axis < 3; ++axis) {
        // centroid range along this axis
        float cmin =  1e20f, cmax = -1e20f;
        for (int i = start; i < end; ++i) {
            float3 c = get_centroid(mesh, i);
            float val = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;
            cmin = std::fmin(cmin, val);
            cmax = std::fmax(cmax, val);
        }
        if (cmax - cmin < 1e-6f) continue;

        float bin_size = (cmax - cmin) / NUM_BINS;

        int  bin_count[NUM_BINS] = {};
        AABB bin_aabb[NUM_BINS];

        for (int i = start; i < end; ++i) {
            float3 c = get_centroid(mesh, i);
            float val = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;
            int bin = (int)((val - cmin) / bin_size);
            if (bin >= NUM_BINS) bin = NUM_BINS - 1;
            bin_count[bin]++;
            bin_aabb[bin].fit(mesh.v0[i]);
            bin_aabb[bin].fit(mesh.v1[i]);
            bin_aabb[bin].fit(mesh.v2[i]);
        }

        // prefix (left→right) and suffix (right→left) sweeps
        AABB left_aabb[NUM_BINS], right_aabb[NUM_BINS];
        int  left_count[NUM_BINS], right_count[NUM_BINS];

        {
            AABB accum; bool accum_valid = false; int accum_count = 0;
            for (int b = 0; b < NUM_BINS; ++b) {
                accum_count += bin_count[b];
                if (bin_count[b] > 0) {
                    accum = accum_valid ? merge_aabb(accum, bin_aabb[b]) : bin_aabb[b];
                    accum_valid = true;
                }
                left_aabb[b]  = accum;
                left_count[b] = accum_count;
            }
        }
        {
            AABB accum; bool accum_valid = false; int accum_count = 0;
            for (int b = NUM_BINS - 1; b >= 0; --b) {
                accum_count += bin_count[b];
                if (bin_count[b] > 0) {
                    accum = accum_valid ? merge_aabb(accum, bin_aabb[b]) : bin_aabb[b];
                    accum_valid = true;
                }
                right_aabb[b]  = accum;
                right_count[b] = accum_count;
            }
        }

        // evaluate split at each bin boundary (between bin b and b+1)
        for (int b = 0; b < NUM_BINS - 1; ++b) {
            int lc = left_count[b];
            int rc = right_count[b + 1];
            if (lc == 0 || rc == 0) continue;
            float cost = (aabb_surface_area(left_aabb[b])  / parent_sa) * lc
                       + (aabb_surface_area(right_aabb[b+1]) / parent_sa) * rc;
            if (cost < best_cost) {
                best_cost  = cost;
                best_axis  = axis;
                best_split = cmin + (b + 1) * bin_size;
            }
        }
    }

    // No split better than the leaf cost — keep this node as a leaf.
    if (best_axis == -1) return;

    // Partition triangles around best_split on best_axis.
    int i = start, j = end - 1;
    while (i <= j) {
        float3 c = get_centroid(mesh, i);
        float val = (best_axis == 0) ? c.x : (best_axis == 1) ? c.y : c.z;
        if (val < best_split) {
            i++;
        } else {
            std::swap(mesh.v0[i],      mesh.v0[j]);
            std::swap(mesh.v1[i],      mesh.v1[j]);
            std::swap(mesh.v2[i],      mesh.v2[j]);
            std::swap(mesh.normals[i], mesh.normals[j]);
            if (!mesh.material_ids.empty())
                std::swap(mesh.material_ids[i], mesh.material_ids[j]);
            j--;
        }
    }
    int split_index = i;
    if (split_index == start || split_index == end) return;

    // Allocate child nodes — do this AFTER partitioning, and read parent
    // fields from local vars (the push_back may reallocate the vector).
    int left_idx  = (int)mesh.bvh_nodes.size();
    mesh.bvh_nodes.push_back(BVHNode());
    int right_idx = (int)mesh.bvh_nodes.size();
    mesh.bvh_nodes.push_back(BVHNode());

    mesh.bvh_nodes[left_idx].start  = start;
    mesh.bvh_nodes[left_idx].end    = split_index;
    mesh.bvh_nodes[right_idx].start = split_index;
    mesh.bvh_nodes[right_idx].end   = end;

    mesh.bvh_nodes[node_idx].left_node_index  = left_idx;
    mesh.bvh_nodes[node_idx].right_node_index = right_idx;

    update_node_bounds(left_idx,  mesh);
    update_node_bounds(right_idx, mesh);

    subdivide(left_idx,  mesh);
    subdivide(right_idx, mesh);
}

// Build the BVH for the mesh so when we raytrace, we don't have to check every triangle.
inline void build_bvh(MeshData& mesh) {
    if (mesh.num_triangles == 0) return;

    std::cout << "Building BVH for " << mesh.num_triangles << " triangles..." << std::endl;
    mesh.bvh_nodes.clear();
    BVHNode root;
    root.start = 0;
    root.end = mesh.num_triangles;
    root.left_node_index  = -1;
    root.right_node_index = -1;
    mesh.bvh_nodes.push_back(root);
    update_node_bounds(0, mesh);
    subdivide(0, mesh);
    std::cout << "BVH Built. Total Nodes: " << mesh.bvh_nodes.size() << std::endl;
}

// Dependency-free OBJ loader with group ('g') parsing.
inline MeshData load_obj(const std::string& filename) {
    MeshData mesh;
    std::vector<float3> temp_vertices;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open mesh file " << filename << std::endl;
        exit(1);
    }

    // current group — default name before any 'g' directive
    std::string current_group = "default";
    int current_group_idx = 0;
    mesh.group_names.push_back(current_group);

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "g") {
            std::string gname;
            ss >> gname;
            if (gname.empty()) gname = "default";
            // find or insert group
            auto it = std::find(mesh.group_names.begin(), mesh.group_names.end(), gname);
            if (it == mesh.group_names.end()) {
                current_group_idx = (int)mesh.group_names.size();
                mesh.group_names.push_back(gname);
            } else {
                current_group_idx = (int)(it - mesh.group_names.begin());
            }
            current_group = gname;
        }
        else if (prefix == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            temp_vertices.push_back(make_float3(x, y, z));
        }
        else if (prefix == "f") {
            // basic face parsing: "f v1 v2 v3" or "f v1//n1 v2//n2 v3//n3" etc.
            std::string segment;
            int v_indices[3];
            int i = 0;
            while (ss >> segment && i < 3) {
                size_t slash = segment.find('/');
                std::string v_str = (slash != std::string::npos) ? segment.substr(0, slash) : segment;
                v_indices[i] = std::stoi(v_str) - 1; // OBJ is 1-indexed
                i++;
            }

            if (i == 3) {
                if(v_indices[0] < (int)temp_vertices.size() &&
                   v_indices[1] < (int)temp_vertices.size() &&
                   v_indices[2] < (int)temp_vertices.size())
                {
                    float3 p0 = temp_vertices[v_indices[0]];
                    float3 p1 = temp_vertices[v_indices[1]];
                    float3 p2 = temp_vertices[v_indices[2]];

                    mesh.v0.push_back(p0);
                    mesh.v1.push_back(p1);
                    mesh.v2.push_back(p2);
                    mesh.num_triangles++;
                    mesh.material_ids.push_back(current_group_idx);

                    // face normal
                    float3 e1 = p1 - p0;
                    float3 e2 = p2 - p0;
                    float3 n;
                    n.x = e1.y * e2.z - e1.z * e2.y;
                    n.y = e1.z * e2.x - e1.x * e2.z;
                    n.z = e1.x * e2.y - e1.y * e2.x;
                    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
                    if (len > 1e-6f) {
                        n.x /= len; n.y /= len; n.z /= len;
                    }
                    mesh.normals.push_back(n);
                }
            }
        }
    }

    std::cout << "Loaded " << mesh.num_triangles << " triangles from " << filename;
    if (mesh.group_names.size() > 1 || mesh.group_names[0] != "default") {
        std::cout << " (" << mesh.group_names.size() << " groups)";
    }
    std::cout << std::endl;
    return mesh;
}

#endif
