import numpy as np

import common
from sample_point_cloud import downsample_point_cloud
from get_neighbours import get_neighbours
from child_node_occupancy import get_child_node


def get_uncles_and_kids(v_origin, s, molecule, denominator):
    uncles = [np.array([]).astype(int) for row in range(7)]
    kids = [np.array([]).astype(int) for row in range(7)]
    cube = common.mesh_cube(np.arange(-1, 2))

    for i in range(1):
        v_neighbour = np.add(v_origin, cube[i])
        v_down, _ = downsample_point_cloud(v_neighbour, np.multiply(v_neighbour, 0), s)
        print(v_down)
        alpha = get_neighbours(v_down, 1)
        beta = get_child_node(v_down, v_neighbour, s, molecule, denominator)

        remainder, x_channel = common.get_parameter(s, molecule, denominator)
        remainder_array = np.mod(v_down, denominator)
        num_child, num_child_label = common.get_child_label(remainder_array, remainder, x_channel)
        child_case, children = common.get_cell(num_child_label)
        print(alpha.shape)
        print(beta.shape)
        print(alpha)
        print(beta)
        for j in range(7):
            uncles[j] = np.append(uncles[j], alpha[child_case[j + 1]])
            kids[j] = np.append(kids[j], beta[child_case[j + 1]])

    return uncles, kids


def build_lut(v_origin, s, molecule, denominator):
    lut = [np.array([]).astype(int) for row in range(7)]
    uncles, kids = get_uncles_and_kids(v_origin, s, molecule, denominator)

    for i in range(7):
        if np.any(kids[i] != 0):
            table = np.concatenate((uncles[i].reshape(-1, 1), kids[i].reshape(-1, 1)), axis=1)
            sort_index = np.argsort(table[:, 0], axis=0)
            sort_table = table[sort_index]

            unique_uncles, unique_index, repeat_counts = np.unique(uncles[i],
                                                                   return_inverse=True,
                                                                   return_counts=True)
            is_single = np.isin(sort_table[:, 0], unique_uncles[np.where(repeat_counts == 1)])
            lut[i] = sort_table[is_single]
            print(unique_index.shape)