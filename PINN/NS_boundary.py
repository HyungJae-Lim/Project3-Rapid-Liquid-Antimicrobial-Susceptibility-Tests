# Setting boundary conditions
# PINN for incompressible 2-D NS equation (steady version)
# coded by St.Watermelon

import tensorflow as tf
import numpy as np


# remove collocation points inside the cylinder
def remove_pt_inside_cyl(xy_col, xc, yc, r):
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in xy_col])
    return xy_col[dst > r, :]

# define boundary condition
def func_u0(y):
    return 4.0 * y * (0.4 - y) / (0.4 ** 2)


def flow_data():
    # set number of data points
    N_b = 200    # inlet and outlet boundary
    N_w = 400    # wall boundary
    N_s = 200    # surface boundary
    N_c = 40000  # collocation points
    N_r = 10000  # additional refining points

    # set boundary
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 0.4
    r = 0.05
    xc = 0.2
    yc = 0.2

    # inlet boundary data, v=0
    inlet_xy = np.linspace([xmin, ymin], [xmin, ymax], N_b)
    inlet_u = func_u0(inlet_xy[:, 1]).reshape(-1,1)
    inlet_v = np.zeros((N_b,1))
    inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

    # outlet boundary condition, p=0
    outlet_xy = np.linspace([xmax, ymin], [xmax, ymax], N_b)
    outlet_p = np.zeros((N_b, 1))

    # wall boundary condition, u=v=0
    wallup_xy = np.linspace([xmin, ymax], [xmax, ymax], N_w)
    walldn_xy = np.linspace([xmin, ymin], [xmax, ymin], N_w)
    wallup_uv = np.zeros((N_w, 2))
    walldn_uv = np.zeros((N_w, 2))

    # cylinder surface, u=v=0
    theta = np.linspace(0.0, 2 * np.pi, N_s)
    cyld_x = (r * np.cos(theta) + xc).reshape(-1, 1)
    cyld_y = (r * np.sin(theta) + yc).reshape(-1, 1)
    cyld_xy = np.concatenate([cyld_x, cyld_y], axis=1)
    cyld_uv = np.zeros((N_s, 2))

    # all boundary conditions except outlet
    xy_bnd = np.concatenate([inlet_xy, wallup_xy, walldn_xy, cyld_xy], axis=0)
    uv_bnd_sol = np.concatenate([inlet_uv, wallup_uv, walldn_uv, cyld_uv], axis=0)

    # collocation points
    x_col = np.random.uniform(xmin, xmax, [N_c, 1])
    y_col = np.random.uniform(ymin, ymax, [N_c, 1])
    xy_col = np.concatenate([x_col, y_col], axis=1)
    # refine points around cylider
    x_col_refine = np.random.uniform(xc-2*r, xc+2*r, [N_r, 1])
    y_col_refine = np.random.uniform(yc-2*r, yc+2*r, [N_r, 1])
    xy_col_refine = np.concatenate([x_col_refine, y_col_refine], axis=1)
    xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

    # remove collocation points inside the cylinder
    xy_col = remove_pt_inside_cyl(xy_col, xc=xc, yc=yc, r=r)

    # concatenation of all boundary and collocation points
    xy_col = np.concatenate([xy_col, xy_bnd, outlet_xy], axis=0)

    # convert all to tensors
    xy_col = tf.convert_to_tensor(xy_col, dtype=tf.float32)
    xy_bnd = tf.convert_to_tensor(xy_bnd, dtype=tf.float32)
    uv_bnd_sol = tf.convert_to_tensor(uv_bnd_sol, dtype=tf.float32)
    outlet_xy = tf.convert_to_tensor(outlet_xy, dtype=tf.float32)
    outlet_p = tf.convert_to_tensor(outlet_p, dtype=tf.float32)

    return xy_col, xy_bnd, uv_bnd_sol, outlet_xy, outlet_p