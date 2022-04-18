import numpy as np
import open3d
import common
from get_neighbours import get_neighbours
from child_node_occupancy import get_child_node
from sample_point_cloud import downsample_point_cloud
from build_lut import build_lut

from lut_sr_fractional import get_super_resolution_v

DATASET_DIR = "../downsample/"
FILE_NAME = "soldier_vox10_0600d1.5"
ORIGIN_FILE = DATASET_DIR + FILE_NAME + ".ply"
RESULT_FILE = "../upsample/" + FILE_NAME + "u2.0.ply"


def main():
    pcd = open3d.io.read_point_cloud(ORIGIN_FILE)
    v_origin = np.asarray(pcd.points)

    s, molecule, denominator = 2.0, 2, 1

    '''
    v_d = np.subtract(v_origin, edge)
    
    v_d = np.around(np.divide(v_d, s))

    remainder, x_channel = common.get_parameter(s, molecule, denominator)
    remainder_array = np.mod(v_d, denominator)
    num_child, num_child_label = common.get_child_label(remainder_array, remainder, x_channel)

    child_case, children = common.get_cell(num_child_label)

    print(num_child_label)
    print(child_case[1].shape)
    print(child_case[1])
    print(num_child[child_case[1]].shape)

    table = np.kron(num_child[child_case[1]], np.ones((children[1].shape[0], 1)))
    print(table.shape)
    get_neighbours(v_d, 1)
    '''
    edge = np.min(v_origin, axis=0)
    v_origin = np.subtract(v_origin, edge)

    v_down = np.floor(np.add(np.divide(v_origin, s), 0.5))

    v_result = get_super_resolution_v(v_down, s, molecule, denominator)
    v_result = np.add(v_result, edge).astype(int)
    c_result = np.zeros(v_result.shape).astype(np.uint8)
    common.write_file(RESULT_FILE, v_result, c_result)


if __name__ == '__main__':
    main()

