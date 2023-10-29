import numpy as np
import open3d as o3d

def show_triangle_mesh_clusters(mesh_path, cluster_path):
    mesh_gt = o3d.io.read_triangle_mesh(mesh_path)
    triangle_clusters, cluster_n_triangles, _ = load_connected_components(
        cluster_path)
    get_the_n_largest_cluster(
        triangle_clusters, cluster_n_triangles, mesh_gt, 10)
    o3d.visualization.draw_geometries([mesh_gt])

def save_connected_components(out_path, mesh):
    print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    print('Save clusters')
    with open(out_path, 'wb') as f:
        np.save(f, triangle_clusters)
        np.save(f, cluster_n_triangles)
        np.save(f, cluster_area)
    return triangle_clusters, cluster_n_triangles, cluster_area


def load_connected_components(out_path):
    print('loading components')
    with open(out_path, 'rb') as f:
        triangle_clusters = np.load(f)
        cluster_n_triangles = np.load(f)
        cluster_area = np.load(f)
    return triangle_clusters, cluster_n_triangles, cluster_area


def get_the_largest_cluster(triangle_clusters, cluster_n_triangles, mesh):
    print("Show largest cluster")
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)


def get_the_n_largest_cluster(triangle_clusters: np.ndarray, cluster_n_triangles: np.ndarray, mesh, n=1):
    print("Show largest cluster")
    cluster_sizes = cluster_n_triangles[triangle_clusters]
    cluster_n_triangles.sort()
    largest_n_cluster_sizes = cluster_n_triangles[-n:]
    print(largest_n_cluster_sizes)
    triangles_to_remove = np.logical_not(
        np.isin(cluster_sizes, largest_n_cluster_sizes))
    mesh.remove_triangles_by_mask(triangles_to_remove)