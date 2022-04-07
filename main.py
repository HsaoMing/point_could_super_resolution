import numpy as np

import common
from get_neighbours import get_neighbours
from child_node_occupancy import get_child_node
from sample_point_cloud import downsample_point_cloud
from build_lut import build_lut
from lut_sr_fractional import get_super_resolution_v

DATASET_DIR = "../dataset/"
ORIGIN_FILE = DATASET_DIR + "downsample.ply"


def main():
    v_down, c_down = common.read_file(ORIGIN_FILE)

    s, molecule, denominator = 1.8, 9, 5
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

    v_result = get_super_resolution_v(v_down, s, molecule, denominator)
    c_result = np.zeros(v_result.shape).astype(np.uint8)
    common.write_file('test.ply', v_result, c_result)


if __name__ == '__main__':
    main()

