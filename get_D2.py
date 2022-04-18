
import open3d
import numpy as np
import torch


DATASET_DIR = "../upsample/"
FILE_NAME = "soldier_vox10_0600d1.5u2.0"
ORIGIN_FILE = DATASET_DIR + FILE_NAME + ".ply"
RESULT_FILE = "../result/" + FILE_NAME + ".ply"


def load_ply_data(filename):
    '''
    load data from ply file.
    '''
    f = open(filename)
    # 1.read all points
    points = []
    for line in f:
        # only x,y,z
        wordslist = line.split(' ')
        try:
            x, y, z = float(wordslist[0]), float(wordslist[1]), float(wordslist[2])
        except ValueError:
            continue
        points.append([x, y, z])
    points = np.array(points)
    points = points.astype(np.int32)  # np.uint8
    # print(filename,'\n','length:',points.shape)
    f.close()

    return points

if __name__ == '__main__':
    filename = ORIGIN_FILE
    pcd = open3d.io.read_point_cloud(ORIGIN_FILE)
    point_cloud = np.asarray(pcd.points)

    data = torch.from_numpy(point_cloud)
    # data = data.unsqueeze(0)
    data = torch.tensor(data).to(torch.float32)

    ori_pcd = open3d.geometry.PointCloud()  # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(data))  # 定义点云坐标位置[N,3]
    ori_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))  # 计算normal
    ori_pcd.paint_uniform_color([0, 0, 0])
    orifile = RESULT_FILE
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 将ply文件中normal类型double转为float32
    lines = open(orifile).readlines()
    to_be_modified = [4,5,6,7,8,9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double', 'float')
    file = open(orifile, 'w')
    # 可视化点云,only xyz
    # open3d.visualization.draw_geometries([ori_pcd])
    for line in lines:
        file.write(line)
    file.close()

