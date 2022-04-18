import numpy as np
import common

from ismember import ismember
from get_neighbours import get_neighbours
from build_lut import build_lut
from sample_point_cloud import upsample_geometry_point_cloud


def get_super_resolution_v(v_down, s, molecule, denominator):
    remainder, x_channel = common.get_parameter(s, molecule, denominator)
    remainder_array = np.mod(v_down, denominator)
    num_child, num_child_label = common.get_child_label(remainder_array, remainder, x_channel)
    child_case, children = common.get_cell(num_child_label)
    cube = common.mesh_cube([0, 1])

    uncles = get_neighbours(v_down, 1)
    lut = build_lut(v_down, s, molecule, denominator)

    v_super = [np.array([]).astype(int) for row in range(8)]
    v_result = np.array([np.nan, np.nan, np.nan]).reshape(-1, 3)

    for i in range(8):
        if np.any(child_case[i] != 0):
            uncles_i = uncles[child_case[i]]
            maximal = np.sum(common.children_decimal(children[i]))
            table = np.multiply(np.ones(uncles_i.shape), maximal)

            if i == 0:
                v_super[i] = upsample_geometry_point_cloud(v_down[child_case[i]], s, children[i], table)
                v_result = v_super[i]
            else:
                is_lut, index = ismember(uncles_i, lut[i - 1][:, 0])
                table[is_lut] = lut[i - 1][index, 1]
                table = np.fliplr(np.unpackbits(table.astype(np.uint8), axis=1)).reshape(-1, 1)
                table = table.astype(float)
                table[np.where(table == 0)] = np.nan
                table = np.multiply(table, np.kron(num_child[child_case[i]], np.ones((cube.shape[0], 1))))
                v_super[i] = upsample_geometry_point_cloud(v_down[child_case[i]], s, cube, table)

                v_result = np.delete(v_result, np.all(np.isnan(v_result), axis=1), axis=0)
                v_result = np.concatenate((v_result, v_super[i]), axis=0)
    v_result = np.unique(v_result, axis=0)

    return v_result
