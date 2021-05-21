# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, stone.py
#

import random
from typing import List

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

from .utils import set_axes_equal
from .math_utils import order_clockwise, pca, rot_matrix, RotationTranslation, Transformation

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

        self.vertices = np.array([a, b, c, d, e, f, g, h])

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

        # Plot the points (with Poly3DCollection, the extents of the plot is not calculate
        ax.plot3D(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 'w.', markersize=1)
        # triangulation
        col = Poly3DCollection(self.triangles, linewidths=1, edgecolors='grey', alpha=.1)
        col.set_facecolor('grey')
        ax.add_collection3d(col)

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
    A stone is described as a collection of vertices (n, 3), initially aligned to the
    coordinate axis length in x-direction, width in y-direction, height in z-direction
    """
    name: str = None

    # geometry: vertices and triangles
    vertices: np.ndarray = None  # (n, 3)
    triangles_index: List[List[int]] = None   # indices to the triangle points
    triangles_values: List[List[np.ndarray]] = None  # triangle coordinates

    # geometrical properties
    eigenvalue: np.ndarray = None
    eigenvector: np.ndarray = None  # (3, 3) !!! vectors as column vectors !!!
    center: np.ndarray = None

    # faces of the stone and their properties
    bottom: np.ndarray = None  # vertices of the bottom face
    bottom_center: np.ndarray = None  # center of the bottom face
    bottom_n: np.ndarray = None  # normal of the bottom face
    top: np.ndarray = None  # vertices of the top face
    top_center: np.ndarray = None  # center of the top face
    top_n: np.ndarray = None  # normal of the top face

    height: float

    def __init__(self, vertices: np.ndarray,
                 triangles_index: List[List[int]] = None, name: str = None):
        """
        A stone is initially with its center at [0, 0, 0] and the axis are aligned to the coordinate
        axis: length in x-direction, width in y-direction, height in z-direction

        :param vertices: (n, 3)
        :param triangles_index: if the vertices are already triangulated, list of the triangle points
        :param name: optional name for a stone
        """
        """
        A stone is initially with its center at [0, 0, 0] and the axis are aligned to the coordinate
        axis: length in x-direction, width in y-direction, height in z-direction
        
        :param vertices: (n, 3)
        :param name: optional name for a stone
        """
        if name:
            self.name = name
        else:
            self.name = random.choice(NAMES)

        # Array of shape (n, 3)
        if vertices.shape[1] != 3:
            raise ValueError('Input must be of shape (n, 3)')

        self.vertices = vertices
        self.eigenvalue, self.eigenvector = self.pca(vertices)

        order = np.argsort(self.eigenvalue)[::-1]  # decreasing from strongest to weakest
        r = rot_matrix(self.eigenvector, order=order)

        # Transform the stone to the origin and align the axis
        center = self.vertices.mean(axis=0)
        t = RotationTranslation(rotation=r, center=center, translation=-center)
        self.vertices = t.transform(self.vertices.T).T

        if not triangles_index and len(vertices) == 8:
            # get the triangles by hand for rectangular stones

            # order the vertices clockwise bottom, clockwise top
            # order the points by z-value
            ind = np.argsort(self.vertices[:, 2])

            # get the 4 lowest and 4 highest points
            bottom = self.vertices[ind[:4]]
            top = self.vertices[ind[4:]]
            a, b, c, d = order_clockwise(bottom)
            e, f, g, h = order_clockwise(top)

            # update the vertices
            self.vertices = np.array([a, b, c, d,  # lower 4 vertices
                                      e, f, g, h])  # upper 4 vertices

            # initialize triangles
            self.triangles_values = [[] for _ in range(12)]  # 8 vertices -> 12 triangles
            self.triangles_index = [
                # a c  b    a  d  c]
                [0, 2, 1], [0, 3, 2],  # bottom
                # e f  g]   e  g  h]
                [4, 5, 6], [4, 6, 7],  # top
                # a b  f    a  f  e
                [0, 1, 5], [0, 5, 4],  # front
                # c d  h    c  h  g
                [2, 3, 7], [2, 7, 6],  # back
                # d a  e    d  e  h
                [3, 0, 4], [3, 4, 7],  # left
                # b c  g    b  g  f
                [1, 2, 6], [1, 6, 5],  # right
            ]

            # get the faces an their properties
            self.update_properties()

    def update_properties(self):
        """
        Get important faces and their normals.
        Simplification: a face is a plane for rectangular stones

        :return:
        """
        self.center = self.vertices.mean(axis=0)
        self.eigenvalue, self.eigenvector = self.pca(self.vertices)
        self.bottom = self.vertices[:4]
        self.bottom_center = self.bottom.mean(axis=0)
        self.bottom_n = np.cross(self.bottom[0] - self.bottom[1], self.bottom[2] - self.bottom[0])

        self.top = self.vertices[4:]
        self.top_center = self.top.mean(axis=0)
        self.top_n = np.cross(self.top[1] - self.top[0], self.top[2] - self.top[0])

        self.height = self.top_center[2] - self.bottom_center[2]

        # update triangle values
        self.triangles_values = [[self.vertices[j] for j in t_ind] for t_ind in self.triangles_index]

    @staticmethod
    def pca(vert: np.ndarray):
        """
        Calculates the principal component analysis of vertices (n, 3)
        and returns the eigenvalue and eigenvectors

        :param vert: collection of vertices
        :return: eigenvalues, eigenvectors
        """
        # Transpose the vertices (util functions for (3, n))

        return pca(vert.T)

    def transform(self, transformation: Transformation):
        """
        Transforms the vertices (``self.vertices``)

        :param transformation: Transformation object including the transformation matrix
        :return: Updates the vertices and all properties
        """

        v = transformation.transform(self.vertices.T)
        self.vertices = v.T
        self.update_properties()

    def add_shape_to_ax(self, ax):
        """
        Adds the shape of the stone (triangles) to the plot

        :param ax: pyplot axis
        :return: -
        """

        # Plot the points
        ax.plot3D(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 'g.', markersize=1)
        # Plot the triangles
        col = Poly3DCollection(self.triangles_values, linewidths=0.4, edgecolors='green', alpha=.2)
        col.set_facecolor('green')
        ax.add_collection3d(col)

        set_axes_equal(ax)

    def add_labels_to_ax(self, ax, positive_eigenvec=True):
        """
        Adds the labels and eigenvectors to the plot

        :param ax: pyplot axis
        :param positive_eigenvec: use the sign of the eigenvectors (left-hand oriented ev appear right-hand)
        :return: -
        """

        # The stones are already aligned to the coordinate axis and centered in (0, 0, 0)
        # calculating the mean and placing the eigenvectors at the mean is not necessary
        mean = np.mean(self.vertices, axis=0)

        # Plot the center
        ax.plot3D(mean[0], mean[1], mean[2], 'r.')

        eigenvalues, eigenvectors = self.pca(self.vertices)

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

    def add_plot_to_ax(self, ax, positive_eigenvec=True):

        self.add_shape_to_ax(ax)

        self.add_labels_to_ax(ax, positive_eigenvec=positive_eigenvec)

        return ax

    def plot(self):
        """
        Plot the stone
        :return:
        """

        fig = plt.figure()
        ax = Axes3D(fig)
        self.add_plot_to_ax(ax)

        set_axes_equal(ax)
        plt.show()

    def __repr__(self):
        return f'<Stone(name={self.name}, vertices={len(self.vertices)})>'
