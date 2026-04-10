#!/usr/bin/env python3
"""
Decimate a PLY mesh for use in the browser-based coarse alignment viewer.

Usage:
    python decimate.py scan.ply -o scan_light.ply
    python decimate.py scan.ply -o scan_light.ply -t 200000
"""

import argparse
import sys

import open3d as o3d


def main():
    parser = argparse.ArgumentParser(
        description="Decimate a PLY mesh via quadric simplification."
    )
    parser.add_argument("input", help="Input PLY file")
    parser.add_argument("-o", "--output", required=True, help="Output PLY file")
    parser.add_argument(
        "-t", "--target-triangles", type=int, default=200_000,
        help="Target number of triangles (default: 200000)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    mesh = o3d.io.read_triangle_mesh(args.input, enable_post_processing=True)
    if not mesh.has_vertices():
        sys.exit(f"ERROR: Failed to load mesh: {args.input}")

    n_in = len(mesh.triangles)
    print(f"  {len(mesh.vertices):,} vertices, {n_in:,} triangles")

    target = min(args.target_triangles, n_in)
    if target >= n_in:
        print("Already at or below target — copying as-is.")
    else:
        print(f"Decimating to ~{target:,} triangles...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)

    mesh.compute_vertex_normals()
    print(f"  Result: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    o3d.io.write_triangle_mesh(
        args.output, mesh, write_vertex_colors=True, write_vertex_normals=True,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
