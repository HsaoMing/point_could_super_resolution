import numpy as np
from common import round_array


def downsample_point_cloud(v_origin, c_origin, s):
    """
    Parameters
    ----------
    v_origin: origin coordinate
    c_origin: origin color attribute
    s: Scale factor

    returns
    -------
    v_down: narray
    c_down: narray

    """
    v_down, index_back, counts = np.unique(round_array(v_origin / s),
                                           return_inverse=True,
                                           return_counts=True,
                                           axis=0)
    c_down = np.zeros(v_down.shape)

    for i in range(3):
        # accumarray in matlab
        c_down[:, i] = np.bincount(index_back, c_origin[:, i])

    c_down = np.divide(c_down, np.array(counts).reshape(-1, 1))

    return round_array(v_down), round_array(c_down)


def upsample_geometry_point_cloud(v_down, s, child, table):
    up_scale = np.ones((child.shape[0], 1))
    scale = np.ones((v_down.shape[0], 1))

    difference = table < 0
    v_up = np.floor(np.add(np.kron(np.multiply(v_down, s), up_scale), 0.5))
    tmp_table = np.kron(scale, child)
    v_up = v_up + np.multiply(np.abs(table), tmp_table) - difference
    v_up = np.delete(v_up, np.all(np.isnan(v_up), axis=1), axis=0)
    return v_up
