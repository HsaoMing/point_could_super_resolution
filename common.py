import numpy as np


def mesh_cube(cube_side):
    x, y, z = np.array(np.meshgrid(cube_side, cube_side, cube_side))
    cube = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    index = np.lexsort((cube[:, 2], cube[:, 1], cube[:, 0]))
    sorted_cube = cube[index, :]

    return sorted_cube


def read_file(file_name):
    # read point cloud
    fp = open(file_name, mode='r')
    file_str = fp.read()
    fp.close()

    point_begin_index = file_str.find("end_header\n") + len("end_header\n")
    origin_data_str = file_str[point_begin_index:]

    origin_data_array = origin_data_str.split()
    # list to np.array and string to int
    origin_data_array = np.array(origin_data_array).astype(float).astype(int)

    origin_data_array = origin_data_array.reshape(-1, 6)
    [v_origin, c_origin] = np.split(origin_data_array, 2, 1)

    return [v_origin, c_origin]


# function from mpeg-gcc-tmc13 octree-raht-ctc-lossy-geom-lossy-attrs.yaml
def position_quantization_scale(rp, gp):
    rp = 6 - rp
    p_min = max(gp - 9, 7)
    start = min(1, gp - (p_min + 6))
    step = max(1, (min(gp - 1, p_min + 7) - p_min) / 5)
    y = start + round(rp * step)
    div = 1 << (abs(y) + 1)
    molecule = div
    denominator = ((1 - 2 * (y < 0)) % div)
    s = molecule / denominator

    return s, molecule, denominator


def intersect(array_1, array_2):
    array_1 = np.array(array_1, order='C')
    array_2 = np.array(array_2, order='C')
    array_1_view = array_1.view([('', array_1.dtype)] * array_1.shape[1])
    array_2_view = array_2.view([('', array_2.dtype)] * array_2.shape[1])
    _, position_1, position_2 = np.intersect1d(array_1_view, array_2_view, return_indices=True)

    return position_1, position_2


def get_parameter(s, molecule, denominator):
    # Fractional Super-Resolution of Voxelized Point Clouds Page 3 Fig 2
    x = np.arange(0, molecule * np.floor(s))
    xd = np.floor(np.add(np.divide(x, s), 0.5))

    hist, _ = np.histogram(xd, x)
    remainder = np.where(hist == np.ceil(s))
    remainder = np.asarray(remainder).reshape(-1)
    difference = x - np.floor(np.add(np.multiply(xd, s), 0.5))

    # np.isin is equal ismember in matlab
    remainder_index = np.isin(xd, remainder)
    x_channel = np.extract(remainder_index, difference).reshape(-1, np.int(np.ceil(s)))

    if molecule % denominator == 0:
        remainder = 0

    return remainder, x_channel


def get_child_label(remainder_array, remainder, x_channel):
    multiply_child_shape = [np.size(remainder_array, 0), np.size(remainder_array, 1), np.size(remainder)]
    multiply_child = np.zeros(multiply_child_shape)

    for i in range(np.size(remainder)):
        multiply_child[:, :, i] = np.sum(x_channel[i, :]) * (np.isin(remainder_array, remainder[i]))

    num_child = np.sum(multiply_child, 2)
    num_child_label = np.dot(np.abs(num_child), np.array([[4, 2, 1]]).T)

    return num_child, num_child_label


def get_cell(num_child_label):
    # child_case is used for index, by vector
    child_case = [np.array(np.isin(num_child_label, 0)).reshape(-1),
                  np.array(np.isin(num_child_label, 4)).reshape(-1),
                  np.array(np.isin(num_child_label, 2)).reshape(-1),
                  np.array(np.isin(num_child_label, 1)).reshape(-1),
                  np.array(np.isin(num_child_label, 3)).reshape(-1),
                  np.array(np.isin(num_child_label, 5)).reshape(-1),
                  np.array(np.isin(num_child_label, 6)).reshape(-1),
                  np.array(np.isin(num_child_label, 7)).reshape(-1)]

    cube = mesh_cube(np.arange(2))
    children_2 = cube[np.where(np.logical_and(cube[:, 1] == 0, cube[:, 2] == 0))]
    children_4 = cube[np.where(cube[:, 0] == 0)]

    children = [cube[np.where(cube.all() == 0)],
                children_2,
                np.roll(children_2, 1, axis=1),
                np.roll(children_2, 2, axis=1),
                children_4,
                np.roll(children_4, 1, axis=1),
                np.roll(children_4, 2, axis=1),
                cube]

    return child_case, children


def children_decimal(children):
    children_label = np.dot(children, np.array([[4, 2, 1]]).T)
    label_weight = np.power(2, children_label)

    return label_weight


def write_file(file_name, v_result, c_result):
    fp = open(file_name, mode='w')
    fp.write("ply\nformat ascii 1.0\ncomment Nothing to comment\n")
    fp.write("element vertex " + str(v_result.shape[0]) + "\n")
    fp.write("property float x\nproperty float y\nproperty float z\n")
    fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    fp.write("end_header\n")

    result = np.concatenate((v_result, c_result), axis=1)
    for line in result:
        line.tofile(fp, sep=' ', format="%s")
        fp.write("\n")
    fp.close()
