# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, stone.py
#

import random

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from .utils import set_axes_equal
from .math_utils import pca, rot_matrix, transform, transform2origin, X, Y, Z
from typing import List


NAMES = ['Achondrit', 'Adakit', 'Aleurit', 'Alkaligranit', 'Alnöit', 'Alvikit', 'Amphibolit', 'Anatexit', 'Andesit', 'Anhydrit', 'Anorthosit', 'Anthrazit', 'Aplit', 'Arenit', 'Arkose', 'Augengneis', 'Basalt', 'Basanit', 'Bauxit', 'Beforsit', 'Bentonit', 'Bergalith', 'Bimsstein', 'Biolithe', 'Blätterkohle', 'Blauschiefer', 'Bohnerz', 'Braunkohle', 'Brekzie', 'Buntsandstein', 'Bändererz', 'Buchit', 'Cancalit', 'Charnockit', 'Chert', 'Chloritschiefer', 'Chondrit', 'Cipollino', 'Dachschiefer', 'Dacit', 'Diabas', 'Diamiktit', 'Diatomit', 'Diorit', 'Dolerit', 'Dolomit', 'Dunit', 'Ehrwaldit', 'Eisenmeteorit', 'Eisenoolith', 'Eklogit', 'Enderbit', 'Erbsenstein', 'Essexit', 'Evaporit', 'Fanglomerat', 'Faserkohle', 'Felsit', 'Fenit', 'Fettkohle', 'Feuerstein', 'Fladenlava', 'Flammkohle', 'Fleckschiefer', 'Flint', 'Flysch', 'Foidit', 'Fortunit', 'Foyait', 'Fruchtschiefer', 'Fulgurit', 'Gabbro', 'Garbenschiefer', 'Gauteit', 'Gasflammkohle', 'Gaskohle', 'Gips', 'Glanz(braun)kohle', 'Glaukophanschiefer', 'Glimmerschiefer', 'Gneis', 'Granit', 'Granitporphyr', 'Granodiorit', 'Granophyr', 'Granulit', 'Graptolithenschiefer', 'Grauwacke', 'Griffelschiefer', 'Grünschiefer', 'Hälleflinta', 'Halitit', 'Hartbraunkohle', 'Harzburgit', 'Hawaiit', 'Hornblendit', 'Hornfels', 'Hornstein', 'Ignimbrit', 'Impaktit', 'Itakolumit', 'Jacupirangit', 'Jumillit', 'Kakirit', 'Kalisalze', 'Kalksandstein', 'Kalkstein', 'Kalksilikatfels', 'Kalksinter', 'Kalktuff', 'Kalziolith', 'Kännelkohle', 'Kaolin', 'Karbonatit', 'Karstmarmore', 'Kataklasit', 'Kennelkohle', 'Keratophyr', 'Kersantit', 'Khondalit', 'Kieselerde', 'Kieselgur', 'Kieselschiefer', 'Kieselsinter', 'Kimberlit', 'Kissenlava', 'Klingstein', 'Knochenbrekzie', 'Knotenschiefer', 'Kohle', 'Kohleeisenstein', 'Kohlenkalk', 'Kokardenerz', 'Konglomerat', 'Kontaktschiefer', 'Korallenerz', 'Kreide', 'Kuckersit', 'Lamproit', 'Lamprophyr', 'Lapilli', 'Lapislazuli', 'Larvikit', 'Lava', 'Latit', 'Lehm', 'Leptynit', 'Letten', 'Leucitit', 'Lherzolith', 'Lignit', 'Limburgit', 'Listwänit', 'Liparit', 'Liptobiolith', 'Lockergestein', 'Löss', 'Lutit', 'Lydit', 'Madupit', 'Magerkohle', 'Mafitit', 'Mandelstein', 'Manganknollen', 'Marmor', 'Massenkalk', 'Mattkohle', 'Meimechit', 'Melaphyr', 'Melilithit', 'Mergel', 'Mergelschiefer', 'Mergelstein', 'Meteorit', 'Migmatit', 'Mikrogabbro', 'Mikrogranit', 'Minette (Ganggestein)', 'Minette (Erz)', 'Moldavit', 'Monchiquit', 'Monzonit', 'MORB', 'Mugearit', 'Mylonit', 'Nephelinbasalt', 'Nephelinit', 'Nephelinsyenit', 'Norit', 'Obsidian', 'OIB', 'Ölschiefer', 'Oolith', 'Ophicalcit', 'Ophiolith', 'Ophit', 'Orendit', 'Pallasit', 'Pechstein', 'Pantellerit', 'Pegmatit', 'Perlit', 'Peridotit', 'Phonolith', 'Phyllit', 'Pikrit', 'Pläner', 'Polzenit', 'Porphyr', 'Porphyrit', 'Prasinit', 'Pseudotachylit', 'Pyroxenit', 'Quarzit', 'Quarzolith', 'Quarzporphyr', 'Radiolarit', 'Rapakiwi', 'Raseneisenstein', 'Rauhaugit', 'Rhyolith', 'Rodingit', 'Rogenstein', 'Sagvandit', 'Sannait', 'Sandstein', 'Schalstein', 'Schiefer', 'Schwarzpelit', 'Serpentinit', 'Shonkinit', 'Silikat-Perowskit', 'Siltstein', 'Skarn', 'Sonnenbrennerbasalt', 'Sövit', 'Spessartit', 'Spiculit', 'Spilit', 'Steinkohle', 'Steinsalz', 'Steinmeteorit', 'Suevit', 'Syenit', 'Talk-Disthen-Schiefer', 'Tektit', 'Tephrit', 'Teschenit', 'Tachylit', 'Theralith', 'Tholeiit', 'Tonalit', 'Tonschiefer', 'Tonstein', 'Trachyt', 'Travertin', 'Troktolith', 'Trondhjemit', 'Tropfstein', 'Tuffstein', 'Unakit', 'Verit', 'Weißschiefer', 'Websterit', 'Wyomingit']  # noqa


class Wall:
    """
    A wall consists of the placed stones and the boundary
    """

    def __init__(self, boundary: 'Boundary', stones: List['Stone'] = None):
        self.boundary = boundary
        self.stones = stones

        if not self.stones:
            self.stones = []


class Boundary:
    """
    The boundary of the wall (same width at top as at the bottom)
    a, b, c, d: bottom 4 vertices (counterclockwise)
    e, f, g h: top 4 vertices (clockwise
    """

    def __init__(self, x=2., y=0.5, z=1):
        """

        :param x: Length [m]
        :param y: Width [m]
        :param z: Heigth [m]
        """
        self.x = x
        self.y = y
        self.z = z

        a = [0, 0, 0]
        b = [x, 0, 0]
        c = [x, y, 0]
        d = [0, y, 0]
        e = [0, 0, z]
        f = [x, 0, z]
        g = [x, y, z]
        h = [0, y, z]

        self.vertices = [a, b, c, d, e, f, g, h]

        self.bottom = np.array([a, b, c, d])
        self.front = np.array([a, b, f, e])
        self.back = np.array([d, c, g, h])
        self.left = np.array([a, e, h, d])
        self.right = np.array([b, f, g, c])

        self.triangles = [
            [a, b, c], [a, c, d],  # bottom
            [a, b, f], [a, f, e],  # front
            [c, d, h], [c, h, g],  # back
            [d, a, e], [d, e, h],  # left
            [b, c, g], [b, g, f]  # right
        ]

    def add_plot_to_ax(self, ax):
        """
        Adds the boundaries to the plot

        :param ax:
        :return:
        """

        # triangulation
        col = Poly3DCollection(self.triangles, linewidths=1, edgecolors='grey', alpha=.1)
        col.set_facecolor('grey')
        ax.add_collection3d(col)
        # print(self.bottom)
        # print(self.bottom.shape, self.bottom.tolist())

        # ax.plot3D(self.bottom[:, 0], self.bottom[:, 1], self.bottom[:, 2], 'k-')
        # ax.plot3D(self.front[:, 0], self.front[:, 1], self.front[:, 2], 'k-')
        # ax.plot3D(self.back[:, 0], self.back[:, 1], self.back[:, 2], 'k-')
        # ax.plot3D(self.left[:, 0], self.left[:, 1], self.left[:, 2], 'k-')
        # ax.plot3D(self.right[:, 0], self.right[:, 1], self.right[:, 2], 'k-')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        set_axes_equal(ax)

    def plot(self):
        """
        Plots the boundary

        :return:
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        self.add_plot_to_ax(ax)
        plt.show()

    def __repr__(self):
        return f'<Boundary(vertices={self.vertices})>'


class Stone:
    """
    A stone is described as a collection of vertices (n, 3), aligned to the coordinate axis
    length in x-direction, width in y-direction, height in z-direction
    """
    name: str = None
    # original data and orientation
    vertices_orig: np.ndarray = None  # (n, 3) a coordinate triple is a row
    eigenvalue_orig: np.ndarray = None
    eigenvector_orig: np.ndarray = None  # (3, 3) !!! vectors as column vectors !!!
    order: np.ndarray = None  # indexes to height, width, length (decreasing) in the original pca
    # transformed vertices and orientation
    vertices: np.ndarray = None  # (n, 3)
    center: np.ndarray = None

    # sides of the stone and their properties
    bottom: np.ndarray = None  # bottom vertices of the stone
    bottom_n: np.ndarray = None  # direction of the bottom normal (downwards
    top: np.ndarray = None  # top vertices of the stone
    top_n = np.ndarray = None  # direction of the top normal (upwards)
    height: float

    def __init__(self, vertices: np.ndarray, triangles: List = None, name: str = None):
        """

        :param vertices: (n, 3)
        :param name:
        """
        if name:
            self.name = name
        else:
            self.name = random.choice(NAMES)

        # Array of shape (n, 3)
        self.vertices_orig = vertices  # store orig for debugging purposes, discard later
        if vertices.shape[1] != 3:
            raise ValueError('Input must be of shape (n, 3)')

        self.eigenvalue_orig, self.eigenvector_orig = self.pca(vertices)
        self.order = np.argsort(self.eigenvalue_orig)[::-1]  # decreasing from strongest to weakest
        r = rot_matrix(self.eigenvector_orig, order=self.order)

        self.vertices, self.center = self.transform2origin(vertices, r)

        # get the faces an their properties
        # self.get_faces()

        # if not triangles and len(vertices) == 8:  # get the triangles by hand for rectangular stones
        #     a, b, c, d = self.bottom
        #     e, f, g, h = self.top
#
        #     self.triangles = [
        #         [a, c, b], [a, d, c],  # bottom
        #         [e, f, g], [e, g, h],  # top
        #         [a, b, f], [a, f, e],  # front
        #         [c, d, h], [c, h, g],  # back
        #         [d, a, e], [d, e, h],  # left
        #         [b, c, g], [b, g, f]  # right
        #     ]

    def get_faces(self):
        """
        Get important faces and their normal.
        Simplification: a face is a plane for rectangular stones

        :return:
        """

        self.bottom = self.vertices[:4]
        self.bottom_n = np.cross(self.bottom[1] - self.bottom[0], self.bottom[2] - self.bottom[0])

        self.top = self.vertices[4:]
        self.top_n = np.cross(self.top[1] - self.top[0], self.top[2] - self.top[0])

    @staticmethod
    def pca(vert: np.ndarray):
        """
        Calculates the principal component analysis of vertices (n, 3)
        and returns the eigenvalue and eigenvectors

        :param vert: collection of vertices
        :return: eigenvalues, eigenvectors
        """
        # Transpose the vertices (util functions for (3, n)

        return pca(vert.T)

    # Calculate the rotation matrix from eigenvectors (basis) to a new basis (coordinate axis)
    # No transposing of data necessary --> use the math_utils function
    # def rot_matrix()

    def transform(self, r=np.array([X, Y, Z]), t=np.array([0, 0, 0])):
        """
        Transforms the vertices (``self.vertices``)

        :param r:
        :param t:
        :return:
        """
        v = transform(self.vertices.T, r, t)
        self.vertices = v.T
        self.center = self.vertices.mean(axis=1)

    @staticmethod
    def transform2origin(vert, r):
        """
        Transforms the stone to the origin and aligns the axis with the rotation matrix.

        :param vert: collection of vertices (n, 3)
        :param r: rotation matrix
        :return: Transformed vertices and the (new) center
        """

        vert_rot, center = transform2origin(vert.T, r)
        return vert_rot.T, center

    def add_shape_to_ax(self, ax, orig=False):
        if orig:
            vert = self.vertices_orig
        else:
            vert = self.vertices

        # Plot the points
        ax.plot3D(vert[:, 0], vert[:, 1], vert[:, 2], 'g.')

        set_axes_equal(ax)
        return vert

    def add_labels_to_ax(self, ax, vert, positive_eigenvec=True):

        # The stones are already aligned to the coordinate axis and centered in (0, 0, 0)
        # calculating the mean and placing the eigenvectors at the mean is not necessary
        mean = np.mean(vert, axis=0)

        # Plot the center
        ax.plot3D(mean[0], mean[1], mean[2], 'r.')

        eigenvalues, eigenvectors = self.pca(vert)

        # Plot the three axes of the stone
        for val, vec in zip(eigenvalues, eigenvectors):
            v = val
            cx, cy, cz = mean  # center coordinates

            # components of the eigenvector
            # use positive eigenvectors (e_vec can direct wrong direction (180°))
            if positive_eigenvec:
                x, y, z = np.sign(vec)*vec
            # use the correct direction of eigenvectors
            else:
                x, y, z = vec
            ax.plot3D([cx, cx + v*x], [cy, cy + v*y], [cz, cz + v*z], 'r')

            ax.text(cx + v*x, cy + v*y, cz + v*z, np.round(v, 2), 'x')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def add_plot_to_ax(self, ax, orig=False, positive_eigenvec=True):

        vert = self.add_shape_to_ax(ax, orig=orig)

        self.add_labels_to_ax(ax, vert, positive_eigenvec=positive_eigenvec)

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
