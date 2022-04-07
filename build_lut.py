import numpy as np

import common
from sample_point_cloud import downsample_point_cloud
from get_neighbours import get_neighbours
from child_node_occupancy import get_child_node


def get_uncles_and_kids(v_origin, s, molecule, denominator):
    uncles = [np.array([]).astype(int) for row in range(7)]
    kids = [np.array([]).astype(int) for row in range(7)]
    cube = common.mesh_cube(np.arange(-1, 2))

    for i in range(cube.shape[0]):
        v_neighbour = np.add(v_origin, cube[i])
        v_down, _ = downsample_point_cloud(v_neighbour, np.multiply(v_neighbour, 0), s)

        alpha = get_neighbours(v_down, 1)
        beta = get_child_node(v_down, v_neighbour, s, molecule, denominator)

        remainder, x_channel = common.get_parameter(s, molecule, denominator)
        remainder_array = np.mod(v_down, denominator)
        num_child, num_child_label = common.get_child_label(remainder_array,
                                                            remainder,
                                                            x_channel)
        child_case, children = common.get_cell(num_child_label)
        for j in range(7):
            uncles[j] = np.append(uncles[j], alpha[child_case[j + 1]])
            kids[j] = np.append(kids[j], beta[child_case[j + 1]])

    return uncles, kids


def build_lut(v_origin, s, molecule, denominator):
    lut = [np.array([]).astype(int) for row in range(7)]
    uncles, kids = get_uncles_and_kids(v_origin, s, molecule, denominator)

    for i in range(7):
        if np.any(kids[i] != 0):
            table = np.concatenate((uncles[i].reshape(-1, 1), kids[i].reshape(-1, 1)),
                                   axis=1)
            sort_index = np.argsort(table[:, 0], axis=0)
            sort_table = table[sort_index]

            s_uncles, s_counts = np.unique(sort_table[:, 0],
                                           return_counts=True)
            is_single = np.isin(sort_table[:, 0],
                                s_uncles[np.where(s_counts == 1)])
            lut[i] = sort_table[is_single]
            m_sort_table = sort_table[~is_single]
            m_uncles, m_index, m_counts = np.unique(m_sort_table[:, 0],
                                                    return_inverse=True,
                                                    return_counts=True)
            tmp = m_sort_table[:, 1].astype(np.uint8).reshape(-1, 1)
            kids_bin = np.unpackbits(tmp, axis=1)
            vector = np.zeros((m_uncles.shape[0], 8))
            for bit in range(8):
                vector[:, bit] = np.bincount(m_index, kids_bin[:, bit])
                vector[:, bit] = np.floor(np.add(np.divide(vector[:, bit], m_counts), 0.5))
            vector = vector.astype(int)
            m_lut = np.concatenate((m_uncles.reshape(-1, 1), np.packbits(vector, axis=1)),
                                   axis=1)
            lut[i] = np.concatenate((lut[i], m_lut), axis=0)
            sort_index = np.argsort(lut[i][:, 0], axis=0)
            lut[i] = lut[i][sort_index]

    return lut
