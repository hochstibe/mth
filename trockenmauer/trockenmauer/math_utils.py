# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 01.03.2021, math_utils.py
# Utility functions, coordinates are treated as columns vectors

from typing import List, Union, Tuple
import numpy as np


class Transformation:
    """
    Base class for a transformation with

     * Number of dimensions
     * homogenous transformation matrix
     * function for applying the transformation on a set of points

    """

    hom_transf_matrix: np.ndarray = None

    def __init__(self, n_dim: int = 3):
        """

        :param n_dim: Number of dimensions
        """
        self.n_dim = n_dim

    def build_hom_transf_matrix(self):
        """
        Dummy function to build the homogenous transformation matrix

        :return:
        """

        """Dummy function, replaced in child classes"""
        self.hom_transf_matrix = create_hom_trans_mat(n_dim=self.n_dim)

    def transform(self, points: np.ndarray):
        """
        Dummy function to apply the transformation on a set of points. Replaced in child classes

        :param points: Coordinate tuples (3, n)
        :return: Transformed points (3, n)
        """
        hom_points = convert_2_hom_coord(points, self.n_dim)
        return transform(hom_points, self.hom_transf_matrix, self.n_dim, homogenous=True)


class Translation(Transformation):
    """
    Translation
    """
    def __init__(self, translation: np.ndarray = np.zeros(3), n_dim=3):
        """

        :param translation: Translation, 1D-array
        :param n_dim: Number of dimensions
        """
        super().__init__(n_dim)
        self.translation = translation

        self.build_hom_transf_matrix()

    def build_hom_transf_matrix(self):
        """
        Build the homogenous transformation matrix for the translation

        :return: Store the matrix as an attribute
        """
        self.hom_transf_matrix = create_hom_trans_mat(t=self.translation, n_dim=self.n_dim)


class Rotation(Transformation):
    """Rotation around a given point"""

    def __init__(self, rotation=np.eye(3), center=np.zeros(3), n_dim=3):
        """

        :param rotation: Rotation-matrix
        :param center: Center of the orientation
        :param n_dim: Number of dimensions
        """
        super().__init__(n_dim)
        self.rotation = rotation
        self.center = center

        self.build_hom_transf_matrix()

    def build_hom_transf_matrix(self):
        """
        Build the homogenous transformation matrix for the rotation
        (translation to the center, rotation, translate back).

        :return: Store the matrix as an attribute
        """
        # translate center to [0, 0, 0]
        translation1 = create_hom_trans_mat(t=-self.center, n_dim=self.n_dim)
        # rotate around the center
        rotation = create_hom_trans_mat(r=self.rotation, n_dim=self.n_dim)
        # translate back
        translation2 = create_hom_trans_mat(t=self.center, n_dim=self.n_dim)

        self.hom_transf_matrix = translation2 @ rotation @ translation1


class RotationTranslation(Transformation):
    """
    Rotation around a given points followed by a translation
    """

    def __init__(self, rotation: np.ndarray = np.eye(3), center: np.ndarray = np.zeros(3),
                 translation: np.ndarray = np.zeros(3), n_dim=3):
        """

        :param rotation: Rotation-matrix
        :param center: Center of the orientation
        :param translation: Translation, 1D-array
        :param n_dim: Number of dimensions
        """
        super().__init__(n_dim)
        # Rotation
        self.rotation = Rotation(rotation=rotation, center=center, n_dim=self.n_dim)
        self.translation = Translation(translation=translation, n_dim=self.n_dim)

        self.build_hom_transf_matrix()

    def build_hom_transf_matrix(self):
        """
        Build the homogenous transformation matrix for the rotation and the following translation
        (translation to the center, rotation, translate back, translation).
        :return:
        """
        self.hom_transf_matrix = self.translation.hom_transf_matrix @ self.rotation.hom_transf_matrix


def transform(points, transf_matrix: np.eye(4), n_dim=3, homogenous=True, **kwargs) -> np.ndarray:
    """
    Transforms a group of points (3, n).

    :param points: Points (3, n)
    :param transf_matrix: (homogenous) transformation matrix (4, 4)
    :param n_dim: Number of dimensions
    :param homogenous: Boolean, if the transformation matrix and points are already homogenous
    :param kwargs: kwargs, if the transformation matrix and points are not homogenous
    :return: Points (3, n) (normal coordinates)
    """

    if not homogenous:
        points = convert_2_hom_coord(points, n_dim)
        transf_matrix = create_hom_trans_mat(kwargs['r'], kwargs['t'])

    # Apply the transformation
    transformed_points = transf_matrix @ points

    return transformed_points[:n_dim, :]  # remove the (homogenous) dimension


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
def create_hom_trans_mat(r: np.ndarray = np.eye(3), t: np.ndarray = np.zeros(3),
                         n_dim: int = 3) -> np.ndarray:
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


def tetra_volume(vertices, voxels):
    """
    Calculates the signed volume of each tetrahedron (voxel) and sums the volumes::

                  1      [ax bx cx dx]
        V = sum( --- det [ay by cy dy] )
                  6      [az bz cz dz]
                         [ 1  1  1  1]


    :param vertices: Coordinates (3, n)
    :param voxels:
    :return:
    """

    # for v in voxels:
    #     print(np.abs(np.linalg.det(np.vstack((vertices.T[v].T, np.ones(4))))))
    # vol_abs = sum([np.abs(np.linalg.det(np.vstack((vertices.T[v].T, np.ones(4))))) for v in voxels]) / 6
    # the tetras after generating with tetgen result in negative volumes --> change the order with [::-1]
    vol_sig = np.sum([np.linalg.det(np.vstack((vertices.T[v].T[::-1], np.ones(4)))) for v in voxels]) / 6
    return vol_sig
