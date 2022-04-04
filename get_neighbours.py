import numpy as np
from common import mesh_cube
from common import intersect


def get_neighbours(v, neighbour_size):
    row, column = v.shape
    base = 2 * neighbour_size + 1

    cube = mesh_cube(np.arange(-1 * neighbour_size, neighbour_size + 1))
    mark = [np.power(2, cube.shape[0] - 2), 1]
    neighs = np.zeros([v.shape[0], 1]).astype(np.uint32)

    # column is usually 3, so 'np.power(base, column)' is odd number
    for i in range(np.power(base, column) // 2):
        position_positive, position_negative = intersect(v, np.add(v, cube[i, :]))
        # by symmetry, simplify the calculation
        # high bit signal positive direction
        neighs[position_positive] = neighs[position_positive] + mark[0]
        neighs[position_negative] = neighs[position_negative] + mark[1]
        mark = np.multiply(mark, np.array([0.5, 2]))

    return neighs
