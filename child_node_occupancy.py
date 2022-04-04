import numpy as np
import common

from ismember import ismember
from sample_point_cloud import upsample_geometry_point_cloud


def child_node_assignment(v_up, v_origin, child):
    is_child, _ = ismember(v_up, v_origin, 'rows')
    cow = child.shape[0]
    is_child = is_child.reshape(cow, -1, order='F')
    tmp = np.multiply(is_child, common.children_decimal(child))
    kids = np.sum(tmp, axis=0).reshape(-1, 1)

    return kids


def get_child_node(v_down, v_origin, s, molecule, denominator):
    remainder, x_channel = common.get_parameter(s, molecule, denominator)
    remainder_array = np.mod(v_down, denominator)
    num_child, num_child_label = common.get_child_label(remainder_array, remainder, x_channel)
    child_case, children = common.get_cell(num_child_label)

    table = np.kron(num_child[child_case[0]], np.ones((children[0].shape[0], 1)))
    v_up = upsample_geometry_point_cloud(v_down[child_case[0]], s, children[0], table)
    kids = child_node_assignment(v_up, v_origin, children[0])
    v_tmp = v_down[child_case[0]]

    for i in range(1, 8):
        table = np.kron(num_child[child_case[i]], np.ones((children[i].shape[0], 1)))
        v_up = upsample_geometry_point_cloud(v_down[child_case[i]], s, children[i], table)

        tmp = child_node_assignment(v_up, v_origin, children[i])
        kids = np.append(kids, tmp, axis=0)
        v_tmp = np.append(v_tmp, v_down[child_case[i]], axis=0)

    _, index = ismember(v_down, v_tmp, 'rows')
    kids = kids[index]

    return kids
