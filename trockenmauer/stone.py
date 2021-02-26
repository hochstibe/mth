# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021
# stone.py -

import numpy as np
import matplotlib.pyplot as plt

from .utils import set_axes_equal


class Stone:
    """
    A stone is described as a collection of vertices, aligned to the coordinate axis
    length in x-direction, width in y-direction, height in z-direction
    """

    def __init__(self, vertices: np.ndarray, name: str = ''):
        self.name = name
        # Array of shape (n, 3)
        self.vertices_orig = vertices  # store orig for debugging purposes, discard later
        if vertices.shape[1] != 3:
            raise ValueError('Input must be of shape (n, 3)')

        eigenvalue_orig, eigenvector_orig = self.ordered_pca(vertices)
        r = self.rot_matrix(eigenvector_orig)

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

        ordered_eigenvector = np.vstack((e_vec[order[2]], e_vec[order[1]], e_vec[order[0]]))
        ordered_eigenvalues = np.array([e_val[order[2]], e_val[order[1]], e_val[order[0]]])

        return ordered_eigenvalues, ordered_eigenvector

    # Calculate the rotation matrix
    @staticmethod
    def rot_matrix(e_vec: np.ndarray,
                   x=np.array([1, 0, 0]), y=np.array([0, 1, 0]), z=np.array([0, 0, 1])) -> np.ndarray:
        """
        Calculates the rotation matrix from the three eigenvectors to the coordinate axis.
        It solves the linear matrix equation ``ax=b`` with the (ordered) eigenvectors ``(e1, e2, e3)`` as ``a``,
        the coordinate axis ``(x, y, z)`` as ``b``::

        (e1, e2, e3) * R = (x, y, z)


        :param e_vec: Array of the three (ordered) eigenvectors
        :param x: x-axis: length will be transformed to this axis
        :param y: y-axis: width will be transformed to this axis
        :param z: z-axis: height will be transformed to this axis
        :return: Rotation matrix R (3, 3)
        """
        # solve ax = b
        a = e_vec
        b = np.vstack((x, y, z))

        x = np.linalg.solve(a, b)
        return x

    @staticmethod
    def transform2origin(vert, r):
        """
        Calculate the center of mass of a given set of vertices (uniform mass distribution)
        :param vert: collection of vertices
        :param r: rotation matrix
        :return: Transformed vertices and the center (new) center
        """
        vert_rot = np.matmul(vert, r)
        # calculate the center (mean of each coordinates)
        m = np.mean(vert_rot, axis=0)
        # shift the center to the origin (0, 0, 0)
        return vert_rot - m, np.array([0, 0, 0])

    def add_plot_to_ax(self, ax):
        # Plot the points
        ax.plot3D(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 'g.')
        # ax.plot3D(np.append(u[:, 0], u[0, 0]), np.append(u[:, 1], u[0, 1]), np.append(u[:, 2], u[0, 2]), 'gray')

        # The stones are already aligned to the coordinate axis and centered in (0, 0, 0)
        # calculating the mean and placing the eigenvectors at the mean is not necessary
        mean = np.mean(self.vertices, axis=0)

        # Plot the center
        ax.plot3D(mean[0], mean[1], mean[2], 'r.')

        eigenvalues, eigenvectors = self.pca(self.vertices)

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
