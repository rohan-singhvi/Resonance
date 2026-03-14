import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.io.wavfile
import trimesh
from numba import cuda

IS_SIMULATION = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
if IS_SIMULATION:
    import numpy as xp

    print("RUNNING IN SIMULATION MODE (CPU)")
else:
    import cupy as xp

    print("RUNNING IN REAL CUDA MODE (GPU)")


@cuda.jit(device=True)
def reflect_vector(dx, dy, dz, nx, ny, nz):
    dot = dx * nx + dy * ny + dz * nz
    rx = dx - 2.0 * dot * nx
    ry = dy - 2.0 * dot * ny
    rz = dz - 2.0 * dot * nz
    return rx, ry, rz


@cuda.jit(device=True)
def intersect_sphere(px, py, pz, dx, dy, dz, radius):
    b = 2.0 * (px * dx + py * dy + pz * dz)
    c = (px * px + py * py + pz * pz) - (radius * radius)
    discriminant = b * b - 4.0 * c

    if discriminant < 0.0:
        return 1e20

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    if t1 > 1e-3:
        return t1
    if t2 > 1e-3:
        return t2
    return 1e20


@cuda.jit(device=True)
def check_listener_hit(
    ray_x, ray_y, ray_z, dx, dy, dz, move_dist, lx, ly, lz, l_radius
):
    rx = lx - ray_x
    ry = ly - ray_y
    rz = lz - ray_z
    t_projection = rx * dx + ry * dy + rz * dz

    if t_projection < 0.0 or t_projection > move_dist:
        return -1.0

    cx = ray_x + dx * t_projection
    cy = ray_y + dy * t_projection
    cz = ray_z + dz * t_projection

    dist_sq = (lx - cx) ** 2 + (ly - cy) ** 2 + (lz - cz) ** 2
    if dist_sq < (l_radius * l_radius):
        return t_projection
    return -1.0


@cuda.jit(device=True)
def intersect_triangle(
    px, py, pz, dx, dy, dz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z
):
    epsilon = 1e-6

    e1x, e1y, e1z = v1x - v0x, v1y - v0y, v1z - v0z
    e2x, e2y, e2z = v2x - v0x, v2y - v0y, v2z - v0z

    h_x = dy * e2z - dz * e2y
    h_y = dz * e2x - dx * e2z
    h_z = dx * e2y - dy * e2x

    a = e1x * h_x + e1y * h_y + e1z * h_z

    if a > -epsilon and a < epsilon:
        return 1e20

    f = 1.0 / a
    s_x, s_y, s_z = px - v0x, py - v0y, pz - v0z
    u = f * (s_x * h_x + s_y * h_y + s_z * h_z)

    if u < 0.0 or u > 1.0:
        return 1e20

    q_x = s_y * e1z - s_z * e1y
    q_y = s_z * e1x - s_x * e1z
    q_z = s_x * e1y - s_y * e1x

    v = f * (dx * q_x + dy * q_y + dz * q_z)
    if v < 0.0 or u + v > 1.0:
        return 1e20

    t = f * (e2x * q_x + e2y * q_y + e2z * q_z)

    if t > 1e-3:
        return t
    return 1e20


@cuda.jit
def ray_trace_kernel(
    rays_pos,
    rays_dir,
    hits_log,
    room_dims,
    listener_pos,
    impulse_response,
    room_type,
    mesh_v0,
    mesh_v1,
    mesh_v2,
    mesh_normals,
    num_triangles,
):
    SPEED_OF_SOUND = 343.0
    SAMPLE_RATE = 44100.0
    LISTENER_RADIUS = 0.5

    idx = cuda.grid(1)
    if idx < rays_pos.shape[0]:
        px, py, pz = rays_pos[idx, 0], rays_pos[idx, 1], rays_pos[idx, 2]
        dx, dy, dz = rays_dir[idx, 0], rays_dir[idx, 1], rays_dir[idx, 2]
        lx, ly, lz = listener_pos[0], listener_pos[1], listener_pos[2]

        dist_traveled = 0.0
        energy = 1.0

        for bounce in range(50):
            min_dist = 1e20
            nx, ny, nz = 0.0, 0.0, 0.0

            if room_type == 0:
                room_x, room_y, room_z = room_dims[0], room_dims[1], room_dims[2]

                if dx > 0.0:
                    d = (room_x - px) / dx
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, -1.0, 0.0, 0.0
                elif dx < 0.0:
                    d = (0.0 - px) / dx
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 1.0, 0.0, 0.0

                if dy > 0.0:
                    d = (room_y - py) / dy
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, -1.0, 0.0
                elif dy < 0.0:
                    d = (0.0 - py) / dy
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, 1.0, 0.0

                if dz > 0.0:
                    d = (room_z - pz) / dz
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, 0.0, -1.0
                elif dz < 0.0:
                    d = (0.0 - pz) / dz
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, 0.0, 1.0

            elif room_type == 1:
                radius = room_dims[0]

                if dy < 0.0:
                    d_floor = (0.0 - py) / dy
                    if d_floor < min_dist:
                        min_dist = d_floor
                        nx, ny, nz = 0.0, 1.0, 0.0

                d_sphere = intersect_sphere(px, py, pz, dx, dy, dz, radius)
                if d_sphere < min_dist:
                    min_dist = d_sphere
                    hit_x = px + dx * min_dist
                    hit_y = py + dy * min_dist
                    hit_z = pz + dz * min_dist
                    nx = hit_x / radius
                    ny = hit_y / radius
                    nz = hit_z / radius

            elif room_type == 2:
                for i in range(num_triangles):
                    v0x, v0y, v0z = mesh_v0[i, 0], mesh_v0[i, 1], mesh_v0[i, 2]
                    v1x, v1y, v1z = mesh_v1[i, 0], mesh_v1[i, 1], mesh_v1[i, 2]
                    v2x, v2y, v2z = mesh_v2[i, 0], mesh_v2[i, 1], mesh_v2[i, 2]

                    dist = intersect_triangle(
                        px, py, pz, dx, dy, dz,
                        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z,
                    )

                    if dist < min_dist:
                        min_dist = dist
                        nx = mesh_normals[i, 0]
                        ny = mesh_normals[i, 1]
                        nz = mesh_normals[i, 2]

            if min_dist >= 1e19:
                break

            hit_dist = check_listener_hit(
                px, py, pz, dx, dy, dz, min_dist, lx, ly, lz, LISTENER_RADIUS
            )

            if hit_dist > 0.0:
                total_dist = dist_traveled + hit_dist
                idx_time = int((total_dist / SPEED_OF_SOUND) * SAMPLE_RATE)
                if idx_time < impulse_response.shape[0]:
                    cuda.atomic.add(impulse_response, idx_time, energy)

            move_dist = min_dist + 1e-3
            px += dx * move_dist
            py += dy * move_dist
            pz += dz * move_dist
            dist_traveled += min_dist

            dx, dy, dz = reflect_vector(dx, dy, dz, nx, ny, nz)

            energy *= 0.85
            if energy < 0.001:
                break

        hits_log[idx] = dist_traveled


def parse_vec3(s):
    try:
        return np.array([float(x) for x in s.split(",")], dtype=np.float32)
    except Exception:
        print(f"Error parsing vector: {s}")
        sys.exit(1)


def generate_rays_host(num_rays, source_pos):
    print(f"Generating {num_rays} rays...")
    positions = np.tile(source_pos, (num_rays, 1)).astype(np.float32)
    random_vectors = np.random.normal(size=(num_rays, 3)).astype(np.float32)
    lengths = np.sqrt(np.sum(random_vectors**2, axis=1, keepdims=True))
    directions = random_vectors / lengths
    return positions, directions


def load_mesh_to_gpu(filepath):
    print(f"Loading mesh: {filepath}")
    try:
        mesh = trimesh.load(filepath)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    triangles = mesh.triangles.astype(np.float32)
    normals = mesh.face_normals.astype(np.float32)

    print(f"Mesh stats: {len(triangles)} triangles")

    v0 = np.ascontiguousarray(triangles[:, 0, :])
    v1 = np.ascontiguousarray(triangles[:, 1, :])
    v2 = np.ascontiguousarray(triangles[:, 2, :])
    normals = np.ascontiguousarray(normals)

    d_v0 = cuda.to_device(v0)
    d_v1 = cuda.to_device(v1)
    d_v2 = cuda.to_device(v2)
    d_normals = cuda.to_device(normals)

    return d_v0, d_v1, d_v2, d_normals, len(triangles)


def main():
    parser = argparse.ArgumentParser(description="GPU Acoustic Ray Tracer")
    parser.add_argument(
        "--room", type=str, choices=["shoebox", "dome", "mesh"], default="shoebox"
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        help="Path to .obj/.stl file (required for --room mesh)",
    )
    parser.add_argument("--rays", type=int, default=100_000)
    parser.add_argument("--dims", type=str, default="10,5,8")
    parser.add_argument("--source", type=str, default="2.0,1.5,1.5")
    parser.add_argument("--listener", type=str, default="8.0,1.5,1.5")
    parser.add_argument("--out", type=str, default="room_impulse.wav")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--ir-len", type=float, default=1000.0, dest="ir_len")
    args = parser.parse_args()

    SAMPLE_RATE = args.sr
    IR_LEN = int(args.sr * args.ir_len / 1000.0)
    NUM_RAYS = args.rays
    SOURCE_POS = parse_vec3(args.source)
    LISTENER_POS = parse_vec3(args.listener)
    h_pos, h_dir = generate_rays_host(NUM_RAYS, SOURCE_POS)

    d_pos = cuda.to_device(h_pos)
    d_dir = cuda.to_device(h_dir)
    d_hits = cuda.to_device(np.zeros(NUM_RAYS, dtype=np.float32))
    d_impulse_response = cuda.to_device(xp.zeros(IR_LEN, dtype=np.float32))
    d_listener_pos = cuda.to_device(LISTENER_POS)

    ROOM_TYPE_ID = 0
    d_v0 = cuda.to_device(np.zeros((1, 3), dtype=np.float32))
    d_v1 = cuda.to_device(np.zeros((1, 3), dtype=np.float32))
    d_v2 = cuda.to_device(np.zeros((1, 3), dtype=np.float32))
    d_normals = cuda.to_device(np.zeros((1, 3), dtype=np.float32))
    num_triangles = 0
    ROOM_DIMS = np.zeros(3, dtype=np.float32)

    if args.room == "shoebox":
        ROOM_TYPE_ID = 0
        ROOM_DIMS = parse_vec3(args.dims)
    elif args.room == "dome":
        ROOM_TYPE_ID = 1
        radius = float(args.dims.split(",")[0])
        ROOM_DIMS = np.array([radius, 0.0, 0.0], dtype=np.float32)
    elif args.room == "mesh":
        if not args.mesh_file:
            print("Error: --mesh-file is required for room type 'mesh'")
            sys.exit(1)
        print(
            "Warning: mesh mode in this script uses O(N) brute-force triangle intersection "
            "with no BVH. For large meshes use the C++ binary (acoustic_sim --room mesh) "
            "which has full BVH acceleration.",
            file=sys.stderr,
        )
        ROOM_TYPE_ID = 2
        d_v0, d_v1, d_v2, d_normals, num_triangles = load_mesh_to_gpu(args.mesh_file)

    d_room_dims = cuda.to_device(ROOM_DIMS)

    threads_per_block = 256
    blocks = (NUM_RAYS + (threads_per_block - 1)) // threads_per_block
    print(f"Launching kernel: {blocks} blocks | {args.room}")
    start_time = time.time()

    ray_trace_kernel[blocks, threads_per_block](  # type: ignore
        d_pos, d_dir, d_hits, d_room_dims, d_listener_pos,
        d_impulse_response, ROOM_TYPE_ID,
        d_v0, d_v1, d_v2, d_normals, num_triangles,
    )

    if not IS_SIMULATION:
        cuda.synchronize()

    print(f"Finished in {time.time() - start_time:.4f}s")

    final_ir = d_impulse_response.copy_to_host()
    max_val = np.max(final_ir)
    print(f"Max amplitude: {max_val}")

    if max_val > 0:
        final_ir = final_ir / max_val

    wav_data = (final_ir * 32767).astype(np.int16)
    scipy.io.wavfile.write(args.out, SAMPLE_RATE, wav_data)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
