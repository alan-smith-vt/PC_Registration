#!/usr/bin/env python3
"""
Mesh alignment pipeline for PLY files exported from Artec Studio.

Coarse-aligns two meshes (via manual 4x4 matrix), refines with point-to-plane
ICP on voxel-downsampled point clouds, then applies the final transform to the
full-resolution source mesh and exports a merged PLY.

Usage:
    python align_meshes.py source.ply target.ply -o merged.ply
    python align_meshes.py source.ply target.ply -o merged.ply --initial-transform coarse.txt
    python align_meshes.py source.ply target.ply -o merged.ply --voxel-size 2.0 --icp-threshold 5.0
"""

import argparse
import sys
import time

import numpy as np
import open3d as o3d


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    print(f"Loading mesh: {path}")
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    if not mesh.has_vertices():
        sys.exit(f"ERROR: Failed to load mesh or mesh has no vertices: {path}")
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"  {n_verts:,} vertices, {n_tris:,} triangles, "
          f"colors={'yes' if mesh.has_vertex_colors() else 'no'}, "
          f"normals={'yes' if mesh.has_vertex_normals() else 'no'}")
    return mesh


def mesh_to_pointcloud(mesh: o3d.geometry.TriangleMesh,
                       voxel_size: float) -> o3d.geometry.PointCloud:
    """Convert mesh to a voxel-downsampled point cloud with normals."""
    # Sample points proportional to surface area for uniform coverage
    n_samples = min(len(mesh.vertices), 2_000_000)
    pcd = mesh.sample_points_uniformly(number_of_points=n_samples)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Estimate normals for point-to-plane ICP
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4 if voxel_size > 0 else 5.0,
            max_nn=30,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    print(f"  Downsampled point cloud: {len(pcd.points):,} points")
    return pcd


def load_initial_transform(path: str) -> np.ndarray:
    """Load a 4x4 transformation matrix from a text file.

    Accepts numpy-style text (space or comma delimited, 4 rows x 4 cols).
    """
    T = np.loadtxt(path)
    if T.shape != (4, 4):
        sys.exit(f"ERROR: Transform must be 4x4, got {T.shape}")
    print(f"Loaded initial transform from {path}:")
    print(T)
    return T


def run_icp(source_pcd: o3d.geometry.PointCloud,
            target_pcd: o3d.geometry.PointCloud,
            initial_transform: np.ndarray,
            threshold: float,
            max_iterations: int) -> np.ndarray:
    """Run point-to-plane ICP and return the final 4x4 transform."""
    print(f"\nRunning point-to-plane ICP "
          f"(threshold={threshold}, max_iter={max_iterations})...")
    t0 = time.time()

    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
        ),
    )
    elapsed = time.time() - t0
    print(f"  ICP completed in {elapsed:.1f}s")
    print(f"  Fitness:  {result.fitness:.6f}  "
          f"(fraction of source points with a correspondence)")
    print(f"  RMSE:     {result.inlier_rmse:.6f}")
    print(f"  Final transform:")
    print(result.transformation)
    return np.asarray(result.transformation)


def apply_transform(mesh: o3d.geometry.TriangleMesh,
                    T: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Apply a 4x4 rigid transform to a mesh (in place)."""
    mesh.transform(T)
    return mesh


def merge_meshes(mesh_a: o3d.geometry.TriangleMesh,
                 mesh_b: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Combine two meshes into one (simple concatenation)."""
    merged = mesh_a + mesh_b
    print(f"\nMerged mesh: {len(merged.vertices):,} vertices, "
          f"{len(merged.triangles):,} triangles")
    return merged


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    print(f"Writing merged mesh to: {path}")
    success = o3d.io.write_triangle_mesh(
        path, mesh, write_vertex_colors=True, write_vertex_normals=True
    )
    if not success:
        sys.exit(f"ERROR: Failed to write mesh to {path}")
    print("Done.")


def save_pointcloud(pcd: o3d.geometry.PointCloud, path: str) -> None:
    print(f"Writing point cloud to: {path}")
    success = o3d.io.write_point_cloud(path, pcd)
    if not success:
        sys.exit(f"ERROR: Failed to write point cloud to {path}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Align two PLY meshes: coarse transform + ICP refinement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", help="Path to the source (moving) PLY mesh")
    parser.add_argument("target", help="Path to the target (fixed) PLY mesh")
    parser.add_argument("-o", "--output", default="merged.ply",
                        help="Output path for the merged PLY (default: merged.ply)")
    parser.add_argument("--initial-transform",
                        help="Path to a 4x4 transform matrix (numpy text format). "
                             "If omitted, identity is used (skips coarse alignment).")
    parser.add_argument("--voxel-size", type=float, default=1.0,
                        help="Voxel size for downsampling before ICP (default: 1.0, "
                             "units match your mesh). Set to 0 to skip downsampling.")
    parser.add_argument("--icp-threshold", type=float, default=3.0,
                        help="Max correspondence distance for ICP (default: 3.0)")
    parser.add_argument("--icp-max-iter", type=int, default=200,
                        help="Max ICP iterations (default: 200)")
    parser.add_argument("--no-icp", action="store_true",
                        help="Skip ICP; only apply the initial transform.")
    parser.add_argument("--save-clouds", action="store_true",
                        help="Also save the downsampled point clouds as PLY "
                             "(source_down.ply, target_down.ply) for inspection.")
    parser.add_argument("--save-transform", metavar="PATH",
                        help="Save the final 4x4 transform to a text file.")

    args = parser.parse_args()

    # ── Load meshes ──────────────────────────────────────────────────────
    source_mesh = load_mesh(args.source)
    target_mesh = load_mesh(args.target)

    # Ensure normals exist on the meshes (needed for good ICP + export)
    if not source_mesh.has_vertex_normals():
        print("Computing normals for source mesh...")
        source_mesh.compute_vertex_normals()
    if not target_mesh.has_vertex_normals():
        print("Computing normals for target mesh...")
        target_mesh.compute_vertex_normals()

    # ── Coarse alignment ─────────────────────────────────────────────────
    if args.initial_transform:
        T_init = load_initial_transform(args.initial_transform)
    else:
        print("\nNo initial transform provided — using identity. "
              "If scans are far apart, provide --initial-transform.")
        T_init = np.eye(4)

    # ── ICP refinement ───────────────────────────────────────────────────
    if args.no_icp:
        print("\nSkipping ICP (--no-icp).")
        T_final = T_init
    else:
        print(f"\nPreparing point clouds (voxel_size={args.voxel_size})...")
        source_pcd = mesh_to_pointcloud(source_mesh, args.voxel_size)
        target_pcd = mesh_to_pointcloud(target_mesh, args.voxel_size)

        if args.save_clouds:
            save_pointcloud(source_pcd, "source_down.ply")
            save_pointcloud(target_pcd, "target_down.ply")

        T_final = run_icp(
            source_pcd, target_pcd, T_init,
            threshold=args.icp_threshold,
            max_iterations=args.icp_max_iter,
        )

    # ── Save transform ───────────────────────────────────────────────────
    if args.save_transform:
        np.savetxt(args.save_transform, T_final, fmt="%.10f")
        print(f"Saved final transform to {args.save_transform}")

    # ── Apply to full-res source and merge ───────────────────────────────
    print("\nApplying final transform to full-resolution source mesh...")
    apply_transform(source_mesh, T_final)

    merged = merge_meshes(source_mesh, target_mesh)
    save_mesh(merged, args.output)


if __name__ == "__main__":
    main()
