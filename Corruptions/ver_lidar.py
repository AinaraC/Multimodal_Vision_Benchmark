import open3d as o3d
import numpy as np

def load_points(points_path):
    points = np.fromfile(points_path, dtype=np.float32)
    points = points.reshape((-1, 4))
    points = points[:, :3]
    return points

# Crear una nube de puntos aleatoria
points = load_points('corruption_examples/lidar/fog/severity1.bin')

# Convertir la matriz de puntos a una nube de puntos de Open3D
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualizar la nube de puntos
o3d.visualization.draw_geometries([point_cloud])
