import unittest
import numpy as np

import common
from sample_point_cloud import downsample_point_cloud

DATASET_DIR = "../dataset/"
ORIGIN_FILE = DATASET_DIR + "longdress_vox10_1300.ply"


class TestCommon(unittest.TestCase):
    # common
    def test_mesh_cube(self):
        cube_side = [-1, 0, 1]
        cube = common.mesh_cube(cube_side)

        self.assertEqual(cube.shape, (np.power(cube.shape[1], 3), 3))

    def test_read_file(self):
        [v_origin, c_origin] = common.read_file(ORIGIN_FILE)

        self.assertEqual(v_origin.shape, (857966, 3))
        self.assertEqual(c_origin.shape, (857966, 3))

    def test_position_quantization_scale(self):
        rp, gp = 6, 10
        s, molecule, denominator = common.position_quantization_scale(rp, gp)
        self.assertEqual(s, molecule / denominator)

    def test_intersect(self):
        array_1 = np.arange(300000).reshape(-1, 3)
        array_2 = np.arange(6, 18).reshape(-1, 3)
        position_1, position_2 = common.intersect(array_1, array_2)

        self.assertEqual(position_1.shape, position_2.shape)

    def test_get_parameter(self):
        s, molecule, denominator = 1.8, 9, 5
        remainder, x_channel = common.get_parameter(s, molecule, denominator)

        self.assertEqual(x_channel.shape, (np.size(remainder), np.ceil(s)))

    def test_get_child_label(self):
        [v_origin, c_origin] = common.read_file(ORIGIN_FILE)
        edge = v_origin.min(0)

        s, molecule, denominator = 1.8, 9, 5
        v_d = np.subtract(v_origin, edge)
        v_d = np.around(np.divide(v_d, s))

        remainder, x_channel = common.get_parameter(s, molecule, denominator)
        remainder_array = np.mod(v_d, denominator)
        num_child, num_child_label = common.get_child_label(remainder_array, remainder, x_channel)

        self.assertEqual(np.sum(num_child_label), 5336290)

    def test_children_decimal(self):
        children = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        label_weight = common.children_decimal(children)

        self.assertEqual(np.sum(label_weight), 240)

    def test_downsample_point_cloud(self):
        [v_origin, c_origin] = common.read_file(ORIGIN_FILE)
        s = 1.8
        v_down, c_down = downsample_point_cloud(v_origin, c_origin, s)

        self.assertEqual(v_down.shape, (290647, 3))
        self.assertEqual(c_down.shape, (290647, 3))


if __name__ == '__main__':
    unittest.main()
