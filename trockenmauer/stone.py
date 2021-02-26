# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021
# stone.py -

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from .utils import set_axes_equal


class Stone:
    """
    A stone is described as a collection of vertices, aligned to the coordinate axis
    length in x-direction, width in y-direction, height in z-direction
    """
    name = None
    # original data and orientation
    vertices_orig = None
    eigenvalue_orig = None
    eigenvector_orig = None
    # transformed vertices and orientation
    vertices = None

    def __init__(self, vertices: np.ndarray, name: str = ''):
        self.name = name
        # Array of shape (n, 3)
        self.vertices_orig = vertices  # store orig for debugging purposes, discard later
        if vertices.shape[1] != 3:
            raise ValueError('Input must be of shape (n, 3)')

        self.eigenvalue_orig, self.eigenvector_orig = self.ordered_pca(vertices)
        order = np.argsort(self.eigenvalue_orig)
        r = self.rot_matrix(self.eigenvector_orig)  # , order=order.tolist())

        self.vertices, self.center = self.transform2origin(vertices, r)

    @staticmethod
    def pca(vert):
        """
        Calculates the principal component analysis and returns the eigenvalue and eigenvectors

        :param vert: collection of vertices
        :return:
        """
        cov_matrix = np.cov(vert.T)
        e_val, e_vec = np.linalg.eig(cov_matrix)

        return e_val, e_vec

    def ordered_pca(self, vert):
        """
        Calculates the three main axes of a group of vertices and returns them in descending order:
        length, width, height

        :param vert: collection of vertices
        :return: direction of length, width, height
        """
        e_val, e_vec = self.pca(vert)

        # index of the ordered eigenvalues: from small to big
        order = np.argsort(e_val)
        print('order', order)

        # ordered_eigenvector = np.vstack((e_vec[order[2]], e_vec[order[1]], e_vec[order[0]]))
        ordered_eigenvector = np.array([e_vec[:, order[2]], e_vec[:, order[1]], e_vec[:, order[0]]])
        ordered_eigenvalues = np.array([e_val[order[2]], e_val[order[1]], e_val[order[0]]])

        return ordered_eigenvalues, ordered_eigenvector

    # Calculate the rotation matrix
    def rot_matrix(self, e_vec: np.ndarray, order: List = None,
                   x=np.array([1, 0, 0]), y=np.array([0, 1, 0]), z=np.array([0, 0, 1])) -> np.ndarray:
        """
        Calculates the rotation matrix from the three eigenvectors to the coordinate axis.
        It solves the linear matrix equation ``ax=b`` with the (ordered) eigenvectors ``(e1, e2, e3)`` as ``a``,
        the coordinate axis ``(x, y, z)`` as ``b``::

           R * e_l = x    |(el, ew, ez) (zero (3,3)) (zero (3,3))|    | a |   |x|
           R * e_w = y -> |(zero (3,3)) (el, ew, ez) (zero (3,3))| *  |...| = |y|
           R * e_z = z    |(zero (3,3)) (zero (3,3)) (el, ew, ez)|    | h |   |z|
           (3, 3) * (3, 1) = (3, 1)  -> (9, 9) * (9, 1) = (9, 1)

           reshape x(9, 1) to R(3, 3)



        :param e_vec: Array of the three (ordered) eigenvectors
        :param x: x-axis: length will be transformed to this axis
        :param y: y-axis: width will be transformed to this axis
        :param z: z-axis: height will be transformed to this axis
        :return: Rotation matrix R (3, 3)
        """
        # solve ax = b
        # a = e_vec
        # b = np.vstack((x, y, z))

        # x = np.linalg.solve(a, b)
        # e_val, e_vec = self.pca(self.vertices_orig)
        # order = np.argsort(e_val)
        # print('order of original', order)
        # rotation works to align axes, but the order of axes is not addressed
        # x = e_vec.T  # original eigenvectors --> axis aligned, but wrong orientation
        # rotation (longest aligns x)
        # x = np.array([e_vec[order[2], :], e_vec[order[1], :], e_vec[order[0], :]]).T
        # x = self.eigenvector_orig  # ordered eigenvectors

        zero = np.zeros((3, 3))
        # rearrange the eigenvectors to the order e_length, e_width, e_height
        # print('order', type(order), order)
        if order:
            orig = np.array([e_vec[:, order[2]], e_vec[:, order[1]], e_vec[:, order[0]]]).T
        else:
            orig = e_vec.T
        a = np.block([[orig, zero, zero],
                      [zero, orig, zero],
                      [zero, zero, orig]])
        b = np.concatenate((x, y, z))
        x = np.linalg.solve(a, b)
        x = np.reshape(x, (3, 3)).T
        return x

    @staticmethod
    def transform2origin(vert, r):
        """
        Calculate the center of mass of a given set of vertices (uniform mass distribution)
        :param vert: collection of vertices
        :param r: rotation matrix
        :return: Transformed vertices and the center (new) center
        """
        # vert_rot = np.matmul(vert, r)
        vert_rot = np.matmul(r, vert.T).T
        # calculate the center (mean of each coordinates)
        m = np.mean(vert_rot, axis=0)
        # shift the center to the origin (0, 0, 0)
        return vert_rot - m, np.array([0, 0, 0])

    def add_plot_to_ax(self, ax, orig=False):

        if orig:
            vert = self.vertices_orig
        else:
            vert = self.vertices
        # Plot the points
        ax.plot3D(vert[:, 0], vert[:, 1], vert[:, 2], 'g.')
        # ax.plot3D(np.append(u[:, 0], u[0, 0]), np.append(u[:, 1], u[0, 1]), np.append(u[:, 2], u[0, 2]), 'gray')

        # The stones are already aligned to the coordinate axis and centered in (0, 0, 0)
        # calculating the mean and placing the eigenvectors at the mean is not necessary
        mean = np.mean(vert, axis=0)

        # Plot the center
        ax.plot3D(mean[0], mean[1], mean[2], 'r.')

        eigenvalues, eigenvectors = self.pca(vert)
        print(np.round(eigenvectors, 1))

        # Plot the three axes of the stone
        for val, vec in zip(eigenvalues, eigenvectors):
            v = np.sqrt(val)
            cx, cy, cz = mean  # center coordinates
            x, y, z = vec  # components of the eigenvector
            ax.plot3D([cx, cx + v*x], [cy, cy + v*y], [cz, cz + v*z], 'r')

            ax.text(cx + v*x, cy + v*y, cz + v*z, np.round(v, 2), 'x')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        set_axes_equal(ax)

        return ax

    def plot(self):
        """
        Plot the stone
        :return:
        """

        # fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.add_plot_to_ax(ax)
        plt.show()

    def __repr__(self):
        return f'<Stone(name={self.name}, vertices={len(self.vertices)})>'
