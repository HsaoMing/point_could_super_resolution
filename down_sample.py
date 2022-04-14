import open3d as o3d

DATASET_DIR = "../dataset/"
FILE_NAME = "soldier_vox10_0600"
ORIGIN_FILE = DATASET_DIR + FILE_NAME + ".ply"
RESULT_FILE = "../downsample/" + FILE_NAME + "x2.ply"

pcd = o3d.io.read_point_cloud(ORIGIN_FILE)
pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, 2)
pcd_new.paint_uniform_color([0, 0, 0])
o3d.io.write_point_cloud(RESULT_FILE, pcd_new)
