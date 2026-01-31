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
};


// we need a fn to populate bvhnodes
// to do this, what if we take the centroid of each triangle
// and use that to build a median-split BVH?
inline float3 get_centroid(const MeshData& mesh, int tri_index) {
    // get vertices of the triangle
    float3 p0 = mesh.v0[tri_index];
    float3 p1 = mesh.v1[tri_index];
    float3 p2 = mesh.v2[tri_index];
    // average the inidividual components
    // to get a "center" point - a centroid
    return make_float3((p0.x + p1.x + p2.x) / 3.0f,
                       (p0.y + p1.y + p2.y) / 3.0f,
                       (p0.z + p1.z + p2.z) / 3.0f);
}

// we need a function to update the bounding box of a node
// this means iterating over all triangles in the node
// and expanding the AABB to include their vertices by using fit()
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

// we need a fn to decide how to split the node
inline void subdivide(int node_idx, MeshData& mesh){
    // first lets check if traingle count is small enough to stop
    if(mesh.bvh_nodes[node_idx].triangle_count() <= 2){
        // stop subdividing as we are at leaf
        return;
    }
    // we need to get the longest axis of the bounding box
    // in order to decide where to split (x, y, or z)
    // i.e. if the box is very wide (X-axis), we should cut it in half along X. 
    // If it is tall (Y-axis), we cut along Y.
    float3 extent = mesh.bvh_nodes[node_idx].bbox.max - mesh.bvh_nodes[node_idx].bbox.min;
    int axis = 0; // Default to X
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.x && extent.z > extent.y) axis = 2;
    // given we now have the axis, we must compute the
    // pivot point - the median along that axis
    // to do this, we must split based on the centroids
    // of the triangles in this node
    float min_c = 1e20f;
    float max_c = -1e20f;
    for(int i = mesh.bvh_nodes[node_idx].start; i < mesh.bvh_nodes[node_idx].end; i++){
        float3 centroid = get_centroid(mesh, i);
        float val = (axis == 0) ? centroid.x : (axis == 1) ? centroid.y : centroid.z;
        min_c = std::fmin(min_c, val);
        max_c = std::fmax(max_c, val);
    }
    float split_pos = (min_c + max_c) * 0.5f;

    // partition triangles based on split_pos
    int i = mesh.bvh_nodes[node_idx].start;
    int j = mesh.bvh_nodes[node_idx].end - 1;
    while(i <= j){
        float3 centroid = get_centroid(mesh, i);
        float val = (axis == 0) ? centroid.x : (axis == 1) ? centroid.y : centroid.z;
        if(val < split_pos){
            i++;
        } else {
            // swap triangles at i and j
            std::swap(mesh.v0[i], mesh.v0[j]);
            std::swap(mesh.v1[i], mesh.v1[j]);
            std::swap(mesh.v2[i], mesh.v2[j]);
            std::swap(mesh.normals[i], mesh.normals[j]);
            j--;
        }
    }
    int split_index = i;

    // (If all centroids are identical, we might get 0 on one side)
    if (split_index == mesh.bvh_nodes[node_idx].start || split_index == mesh.bvh_nodes[node_idx].end) {
        return; // Failed to split (overlapping geometry), make it a leaf.
    }

    // child nodes
    int left_idx = mesh.bvh_nodes.size();
    mesh.bvh_nodes.push_back(BVHNode());
    int right_idx = mesh.bvh_nodes.size();
    mesh.bvh_nodes.push_back(BVHNode());

    // setup left child
    BVHNode& left_node = mesh.bvh_nodes[left_idx];
    left_node.start = mesh.bvh_nodes[node_idx].start;
    left_node.end = split_index;

    // setup right child
    BVHNode& right_node = mesh.bvh_nodes[right_idx];
    right_node.start = split_index;
    right_node.end = mesh.bvh_nodes[node_idx].end;

    // update parent node to point to children
    mesh.bvh_nodes[node_idx].left_node_index = left_idx;
    mesh.bvh_nodes[node_idx].right_node_index = right_idx;

    // update bounds for children
    update_node_bounds(left_idx, mesh);
    update_node_bounds(right_idx, mesh);

    // recursively subdivide children
    subdivide(left_idx, mesh);
    subdivide(right_idx, mesh);
}

// built the BVH for the mesh so when we raytrace, we don't have to check every triangle
inline void build_bvh(MeshData& mesh) {
    if (mesh.num_triangles == 0) return;

    std::cout << "Building BVH for " << mesh.num_triangles << " triangles..." << std::endl;
    // clear the old nodes if any
    mesh.bvh_nodes.clear();
    // create root
    BVHNode root;
    root.start = 0;
    root.end = mesh.num_triangles; // The root owns everything
    root.left_node_index = -1;
    root.right_node_index = -1;
    
    // Add root to the list (it will be index 0)
    mesh.bvh_nodes.push_back(root);
    // we then need to update its bounds
    update_node_bounds(0, mesh);
    // and recursively split it
    subdivide(0, mesh);
    std::cout << "BVH Built. Total Nodes: " << mesh.bvh_nodes.size() << std::endl;
}

// dependency-free OBJ loader
inline MeshData load_obj(const std::string& filename) {
    MeshData mesh;
    std::vector<float3> temp_vertices;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open mesh file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            temp_vertices.push_back(make_float3(x, y, z));
        } 
        else if (prefix == "f") {
            // extremely basic face parsing "f v1 v2 v3" or "f v1//n1 v2//n2 v3//n3"
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
                // check bounds
                if(v_indices[0] < temp_vertices.size() && 
                   v_indices[1] < temp_vertices.size() && 
                   v_indices[2] < temp_vertices.size()) 
                {
                    float3 p0 = temp_vertices[v_indices[0]];
                    float3 p1 = temp_vertices[v_indices[1]];
                    float3 p2 = temp_vertices[v_indices[2]];

                    mesh.v0.push_back(p0);
                    mesh.v1.push_back(p1);
                    mesh.v2.push_back(p2);
                    mesh.num_triangles++;

                    // calculate face normal
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
    
    std::cout << "Loaded " << mesh.num_triangles << " triangles from " << filename << std::endl;
    return mesh;
}

#endif