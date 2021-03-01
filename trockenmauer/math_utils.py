# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 01.03.2021, math_utils.py
#

from typing import List
import numpy as np


# Homogenous transformation functions: adapted from Jonas Meyer
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
    Converts a collection of 3D vectors to homogenous coord vector

    :param t: Coordinates (3, n) or (n, 3)
    :return: Homogenous coordinates (4, n)
    """
    if t.shape[0] != 3 and t.shape[1] == 3:
        t = t.T
    num = t.shape[1]  # number of points

    return np.concatenate((t, np.ones((1, num))), axis=0)


def transform(points, r, t: np.ndarray = np.array([0, 0, 0])):
    """
    Transforms a group of points (3, n) or (n, 3).
    If three points are given, they must be given as vectors (ndim, npoint)
    The translation can be given as a vector.

    :param points:
    :param r:
    :param t:
    :return:
    """
    transpose = False
    if points.shape[0] != 3 and points.shape[1] == 3:
        # Transpose the points from (n, 3) to (3, n)
        transpose = True
        points = points.T

    t = create_hom_trans_mat(r, t)

    transformed_points = t @ points[:3, :]  # remove the (homogenous) dimension

    # Transpose the points to the format of the input points
    if transpose:
        return transformed_points.T
    else:
        return transformed_points


def pca(points: np.ndarray):
    """
    Calculates the principal component analysis of vertices (n, 3)
    and returns the eigenvalue and eigenvectors

    :param points: collection of vertices
    :return: eigenvalues, eigenvectors
    """

    cov_matrix = np.cov(points.T)
    e_val, e_vec = np.linalg.eig(cov_matrix)

    return np.sqrt(e_val), e_vec


def rot_matrix(e_vec: np.ndarray, order: List = (0, 1, 2),
               x: np.ndarray = None, y: np.ndarray = None, z: np.ndarray = None
               ) -> np.ndarray:
    """
    Calculates the rotation matrix from the three eigenvectors
    to the three axis x, y, z.
    If an order of the eigenvectors is given (descending), the strongest
    eigenvector is rotated to the x-axis, the weakest to the z-axis.
    If the rotation is from the eigenvectors to the respective coordinate
    system, the eigenvectors are by definition the rotation matrix.
    For an arbitrary new basis, it solves the linear matrix equation
    ``ax=b`` with the (ordered) eigenvectors ``(e1, e2, e3)`` as ``a``,
    the new basis ``(x, y, z)`` as ``b``::

       R * e_l = x    |(e1, e1, e3) (zero (3,3)) (zero (3,3))|    | a |   |x|
       R * e_w = y -> |(zero (3,3)) (e1, e2, e3) (zero (3,3))| *  |...| = |y|
       R * e_h = z    |(zero (3,3)) (zero (3,3)) (e1, e2, e3)|    | h |   |z|
       (3, 3) * (3, 1) = (3, 1)  -> (9, 9) * (9, 1) = (9, 1)

       reshape x(9, 1) to R(3, 3)

    :param e_vec: Array of the three (ordered) eigenvectors
    :param order: order of the eigenvectors (ascending). With the default values,
                  no ordering is applied
    :param x: x-axis: length will be transformed to this axis; np.array([1, 0, 0])
    :param y: y-axis: width will be transformed to this axis; np.array([0, 1, 0])
    :param z: z-axis: height will be transformed to this axis; np.array([0, 0, 1])
    :return: Rotation matrix R (3, 3)
    """
    # e1x e2x e3x
    # e1y e2y e3y = e_vec
    # e1z e2z e3z
    # Rearrange the eigenvectors according to the given order and transpose
    # if the default ordering is selected (0, 1, 2) orig == e_vec
    e_vec = np.array([e_vec[:, order[0]], e_vec[:, order[1]], e_vec[:, order[2]]])

    if x and y and z:  # Rotation matrix from eigenvectors to a new basis
        e_vec = e_vec.T
        zero = np.zeros((3, 3))
        a = np.block([[e_vec, zero, zero],
                      [zero, e_vec, zero],
                      [zero, zero, e_vec]])
        b = np.concatenate((x, y, z))
        x = np.linalg.solve(a, b)
        x = np.reshape(x, (3, 3)).T  # It's correct with transpose...
    else:  # The eigenvectors are the rotation matrix to the respective coordinate system
        x = e_vec
    return x


def transform2origin(points, r):
    """
    Calculate the center of mass of a given set of vertices (uniform mass distribution)::

       R * v = v' with R = (3, 3), v = (3, n), v' = (3, n)

    :param points: collection of 3D points (n, 3)
    :param r: rotation matrix
    :return: Transformed vertices and the (new) center
    """
    # center of the vertices
    m = np.mean(points, axis=0)

    transform(points, r, m)

    return transform(points, r, -m), np.zeros(3)
