# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 01.03.2021, math_utils.py
# Utility functions, coordinates are treated as columns vectors

from typing import List, Union, Tuple
import numpy as np


# unit vectors
X = np.array([1, 0, 0])
Y = np.array([0, 1, 0])
Z = np.array([0, 0, 1])


def order_clockwise(points: np.ndarray):
    """
    Orders 4 points in a clockwise order (ignoring z-value)
    :param points:
    :return:
    """

    # inspired by https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

    # sort the points based on their x-coordinates
    x_sorted = points[np.argsort(points[:, 0]), :]
    # get the two min- and max-x-coordinate-points
    x_min_2 = x_sorted[:2, :]
    x_max_2 = x_sorted[2:, :]

    # sort the min-x coordinates according to their
    # y-coordinates so we can get the point closest to the origin and the other
    a, d = x_min_2[np.argsort(x_min_2[:, 1]), :]

    # calculate the euclidean distance from a (closest to origin)
    # to the 2 points with max-x-coordinates.
    # The closer point is b, the further c
    distances = np.linalg.norm(a - x_max_2, axis=1)
    b, c = x_max_2[np.argsort(distances), :]

    return a, b, c, d


# Homogenous transformation functions: adapted from Jonas Meyer
def create_hom_trans_mat(r: np.ndarray, t: np.ndarray, n_dim: int = 3) -> np.ndarray:
    """
    Converts a rotation matrix and translation vector as numpy array to a
    homogenous transformation matrix as numpy array.::

       R = (3, 3), t = (3, 1)

             |r1 r2 r3 t1|
       Ht =  |r4 r5 r6 t2|
             |r7 r8 r9 t3|
             | 0  0  0  1|

    :param r: Rotation matrix (3, 3)
    :param t: Translation vector (3, 1) or (3, ) or (1, 3)
    :param n_dim: Number of coordinate dimensions (without the added homogenous dimension)
    :return: Homogenous transformation matrix (4, 4)
    """
    if t.shape != (n_dim, 1):
        t = np.reshape(t, (n_dim, 1))
    row = np.append(np.zeros(n_dim), 1)
    return np.vstack((np.hstack((r, t)), row))


def invert_hom_trans_mat(t: np.ndarray) -> np.ndarray:
    """
    Inverts homogenous transformation matrix

    :param t: Homogenous transformation matrix (4, 4)
    :return: Homogenous transformation matrix (4, 4) to revert the transformation
    """
    if t.shape[0] != t.shape[1]:
        raise ValueError('It has to be a homogenous transformation matrix by shape (4,4)')
    n_dim = t.shape[0] - 1  # Dimension of (normal) coordinates
    r_inv = t[0:n_dim, 0:n_dim].T
    t_inv = (-1 * r_inv) @ t[0:n_dim, n_dim]
    return create_hom_trans_mat(r_inv, t_inv, n_dim)


def convert_2_hom_coord(t: np.ndarray, n_dim: int = 3) -> np.ndarray:
    """
    Converts a collection of 3D vectors to homogenous coord vector

    :param t: Coordinates (3, n)
    :param n_dim: Number of coordinate dimensions
    :return: Homogenous coordinates (4, n)
    """
    if t.shape[0] != n_dim:
        raise ValueError

    num = t.shape[1]  # number of points

    return np.concatenate((t, np.ones((1, num))), axis=0)


def pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the principal component analysis of vertices (3, n)
    and returns the eigenvalue and eigenvectors (as column vectors).

    :param points: collection of vertices
    :return: eigenvalues, eigenvectors
    """

    cov_matrix = np.cov(points)
    e_val, e_vec = np.linalg.eig(cov_matrix)

    return np.sqrt(e_val), e_vec


def rot_matrix(e_vec: np.ndarray, order: Union[List, np.ndarray] = (0, 1, 2),
               x: Union[List] = None, y: Union[List] = None,
               z: Union[List] = None) -> np.ndarray:
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

       R * e_l = x    |(e1, e1, e3).T (zero (3,3)) (zero (3,3))|    | a |   |xx|
       R * e_w = y -> |(zero (3,3)) (e1, e2, e3).T (zero (3,3))| *  |...| = |yx|
       R * e_h = z    |(zero (3,3)) (zero (3,3)) (e1, e2, e3).T|    | h |   |zz|
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
    # e2x e0x e1x                                                    e0x e0y e0z
    # e2y e0y e1y = e_vec (input) -> e_vec_t (ordered, Transposed) = e1x e1y e1z
    # e2z e0z e1z                                                    e2x e2y e2z
    # Rearrange the eigenvectors according to the given order and transpose
    # if the default ordering is selected (0, 1, 2) orig == e_vec
    e_vec_t = e_vec[:, order].T
    # e_vec[:, order[0]] = select all rows of the first column [[e1x], [e1y], [e1z]]

    if x and y and z:  # Rotation matrix from eigenvectors to a new basis
        # R * e_vec = b -> solve for R with three known points with a*x = b
        zero = np.zeros((3, 3))
        a = np.block([[e_vec_t, zero, zero],
                      [zero, e_vec_t, zero],
                      [zero, zero, e_vec_t]])
        b = np.vstack((x, y, z)).T  # b = basis vectors as column vectors: | x y z |
        b = b.reshape(9, 1)  # b = [xx, yx, zx, xy, yy, zy, zx, zy, zz]
        x = np.linalg.solve(a, b)
        x = np.reshape(x, (3, 3))  # It's correct with transpose...
    else:  # The eigenvectors are the rotation matrix to the respective coordinate system
        # https://en.wikipedia.org/wiki/Change_of_basis,
        # https://medium.com/swlh/eigenvalues-and-eigenvectors-5fbc8b037eed
        # x_old = A * x_new -> x_new = A^-1 * x_old --> e_vec.T == e_vec^-1
        x = e_vec_t
    return x


def transform(points: np.ndarray, r: np.ndarray = np.array([X, Y, Z]).T,
              t: np.ndarray = np.array([0, 0, 0]), n_dim: int = 3) -> np.ndarray:
    """
    Transforms a group of points (3, n).
    The translation can be given as a vector.::

       R * (v+t) = v' with R = (3, 3), v = (3, n), t=(3, 1), v' = (3, n)

    :param points: Points (3, n)
    :param r: Rotation matrix (3, 3)
    :param t: optional translation vector (3, 1) or (3, ) or (1, 3)
    :param n_dim: Number of coordinate dimensions
    :return: Points (3, n)
    """
    if points.shape[0] != n_dim:
        raise ValueError

    t = create_hom_trans_mat(r, t, n_dim)
    points = convert_2_hom_coord(points, n_dim)

    transformed_points = t @ points
    transformed_points = transformed_points[:3, :]  # remove the (homogenous) dimension

    return transformed_points


def transform2origin(points: np.ndarray, r: np.ndarray = np.array([X, Y, Z]).T
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the center of mass of a given set of vertices (uniform mass distribution).
    The transformation includes a translation of the center of mass to (0, 0, 0).

    :param points: collection of 3D points (3, n)
    :param r: rotation matrix
    :return: Transformed vertices and the (new) center
    """
    # center of the vertices
    m = np.mean(points, axis=1)

    # transform to origin
    # t = transform(points, t=-m)
    # print('transform to origin', t.mean(axis=1))
    # # rotate
    # t = transform(points, r)
    # print('rotatate', t.mean(axis=1))
    # # transform back
    # t = transform(points, t=m)
    # print('transform back', t.mean(axis=1))

    new = transform(points, r, -m)

    return new, new.mean(axis=1)
