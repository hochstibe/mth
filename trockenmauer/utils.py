# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, utils.py
#

import numpy as np


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Credits: https://stackoverflow.com/a/31364297

    :param ax: a matplotlib axis
    :return: None
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def create_hom_trans_mat(r: np.ndarray, t: np.ndarray):
    """
    Converts a rotation matrix and translation vector as numpy array to a
    homogenous transformation matrix as numpy array

    :param r: Rotation matrix (3, 3)
    :param t: Translation vector (3, 1) or (1, 3)
    :return: Homogenous transformation matrix (4, 4)
    """
    if t.shape != (3, 1):
        t = np.reshape(t, (3, 1))
    return np.concatenate((np.concatenate((r, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)


def invert_hom_trans_mat(t):
    """
    Inverts homogenous transformation matrix

    :param t: Homogenous transformation matrix (4, 4)
    :return: Homogenous transformation matrix (4, 4) to revert the transformation
    """
    if t.shape != (4, 4):
        raise ValueError('It has to be a homogenous transformation matrix by shape (4,4)')
    r_inv = t[0:3, 0:3].T
    t_inv = (-1 * r_inv) @ t[0:3, 3]
    return create_hom_trans_mat(r_inv, t_inv)


def convert_2_hom_coord(t):
    """
    Converts 3D vector to homogenous coord vector

    :param t: Coordinates (3, 1) or (1, 3)
    :return: Homogenous coordinates (4, 1)
    """
    if t.shape != (3, 1):
        t = np.reshape(t, (3, 1))
    return np.concatenate((t, np.array([[1]])), axis=0)
