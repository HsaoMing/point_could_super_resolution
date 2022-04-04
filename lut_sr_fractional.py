import read_file
import numpy as np
import common

DATASET_DIR = "../dataset/"
ORIGIN_FILE = DATASET_DIR + "longdress_vox10_1300.ply"

s, molecule, denominator = 1.8, 9, 5

remainder, x_channel = common.get_parameter(s, molecule, denominator)

[v_origin, c_origin] = read_file.read_file(ORIGIN_FILE)
T = v_origin.min(0)

v_d = np.subtract(v_origin, T)
v_d = np.around(np.divide(v_d, s))
r = np.mod(v_d, denominator)

multiply_child_shape = [np.size(r, 0), np.size(r, 1), np.size(remainder)]
multiply_child = np.zeros(multiply_child_shape)

for i in range(0, np.size(remainder)):
    multiply_child[:, :, i] = np.sum(x_channel[i, :]) * (np.isin(r, remainder[i]))

num_child = np.sum(multiply_child, 2)
num_child_label = np.dot(np.abs(num_child), np.array([[4, 2, 1]]).T)

child_case_1 = np.isin(num_child_label, 0)
child_case_2 = np.isin(num_child_label, 4)
child_case_3 = np.isin(num_child_label, 2)
child_case_4 = np.isin(num_child_label, 1)
child_case_5 = np.isin(num_child_label, 3)
child_case_6 = np.isin(num_child_label, 5)
child_case_7 = np.isin(num_child_label, 6)
child_case_8 = np.isin(num_child_label, 7)

print(sum(child_case_3))

