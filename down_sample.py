import open3d as o3d
import numpy

DATASET_DIR = "../downsample/"
FILE_NAME = "soldier_vox10_0600d2.0"
ORIGIN_FILE = DATASET_DIR + FILE_NAME + ".ply"
RESULT_FILE = "../downsample/" + FILE_NAME + "d2.0.ply"

pcd = o3d.io.read_point_cloud(ORIGIN_FILE)
pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, 4)


o3d.io.write_point_cloud(RESULT_FILE, pcd_new)
