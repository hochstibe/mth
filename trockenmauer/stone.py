# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, stone.py
#

from typing import List
import random

import numpy as np
import matplotlib.pyplot as plt

from .utils import set_axes_equal


NAMES = ['Achondrit', 'Adakit', 'Aleurit', 'Alkaligranit', 'Alnöit', 'Alvikit', 'Amphibolit', 'Anatexit', 'Andesit', 'Anhydrit', 'Anorthosit', 'Anthrazit', 'Aplit', 'Arenit', 'Arkose', 'Augengneis', 'Basalt', 'Basanit', 'Bauxit', 'Beforsit', 'Bentonit', 'Bergalith', 'Bimsstein', 'Biolithe', 'Blätterkohle', 'Blauschiefer', 'Bohnerz', 'Braunkohle', 'Brekzie', 'Buntsandstein', 'Bändererz', 'Buchit', 'Cancalit', 'Charnockit', 'Chert', 'Chloritschiefer', 'Chondrit', 'Cipollino', 'Dachschiefer', 'Dacit', 'Diabas', 'Diamiktit', 'Diatomit', 'Diorit', 'Dolerit', 'Dolomit', 'Dunit', 'Ehrwaldit', 'Eisenmeteorit', 'Eisenoolith', 'Eklogit', 'Enderbit', 'Erbsenstein', 'Essexit', 'Evaporit', 'Fanglomerat', 'Faserkohle', 'Felsit', 'Fenit', 'Fettkohle', 'Feuerstein', 'Fladenlava', 'Flammkohle', 'Fleckschiefer', 'Flint', 'Flysch', 'Foidit', 'Fortunit', 'Foyait', 'Fruchtschiefer', 'Fulgurit', 'Gabbro', 'Garbenschiefer', 'Gauteit', 'Gasflammkohle', 'Gaskohle', 'Gips', 'Glanz(braun)kohle', 'Glaukophanschiefer', 'Glimmerschiefer', 'Gneis', 'Granit', 'Granitporphyr', 'Granodiorit', 'Granophyr', 'Granulit', 'Graptolithenschiefer', 'Grauwacke', 'Griffelschiefer', 'Grünschiefer', 'Hälleflinta', 'Halitit', 'Hartbraunkohle', 'Harzburgit', 'Hawaiit', 'Hornblendit', 'Hornfels', 'Hornstein', 'Ignimbrit', 'Impaktit', 'Itakolumit', 'Jacupirangit', 'Jumillit', 'Kakirit', 'Kalisalze', 'Kalksandstein', 'Kalkstein', 'Kalksilikatfels', 'Kalksinter', 'Kalktuff', 'Kalziolith', 'Kännelkohle', 'Kaolin', 'Karbonatit', 'Karstmarmore', 'Kataklasit', 'Kennelkohle', 'Keratophyr', 'Kersantit', 'Khondalit', 'Kieselerde', 'Kieselgur', 'Kieselschiefer', 'Kieselsinter', 'Kimberlit', 'Kissenlava', 'Klingstein', 'Knochenbrekzie', 'Knotenschiefer', 'Kohle', 'Kohleeisenstein', 'Kohlenkalk', 'Kokardenerz', 'Konglomerat', 'Kontaktschiefer', 'Korallenerz', 'Kreide', 'Kuckersit', 'Lamproit', 'Lamprophyr', 'Lapilli', 'Lapislazuli', 'Larvikit', 'Lava', 'Latit', 'Lehm', 'Leptynit', 'Letten', 'Leucitit', 'Lherzolith', 'Lignit', 'Limburgit', 'Listwänit', 'Liparit', 'Liptobiolith', 'Lockergestein', 'Löss', 'Lutit', 'Lydit', 'Madupit', 'Magerkohle', 'Mafitit', 'Mandelstein', 'Manganknollen', 'Marmor', 'Massenkalk', 'Mattkohle', 'Meimechit', 'Melaphyr', 'Melilithit', 'Mergel', 'Mergelschiefer', 'Mergelstein', 'Meteorit', 'Migmatit', 'Mikrogabbro', 'Mikrogranit', 'Minette (Ganggestein)', 'Minette (Erz)', 'Moldavit', 'Monchiquit', 'Monzonit', 'MORB', 'Mugearit', 'Mylonit', 'Nephelinbasalt', 'Nephelinit', 'Nephelinsyenit', 'Norit', 'Obsidian', 'OIB', 'Ölschiefer', 'Oolith', 'Ophicalcit', 'Ophiolith', 'Ophit', 'Orendit', 'Pallasit', 'Pechstein', 'Pantellerit', 'Pegmatit', 'Perlit', 'Peridotit', 'Phonolith', 'Phyllit', 'Pikrit', 'Pläner', 'Polzenit', 'Porphyr', 'Porphyrit', 'Prasinit', 'Pseudotachylit', 'Pyroxenit', 'Quarzit', 'Quarzolith', 'Quarzporphyr', 'Radiolarit', 'Rapakiwi', 'Raseneisenstein', 'Rauhaugit', 'Rhyolith', 'Rodingit', 'Rogenstein', 'Sagvandit', 'Sannait', 'Sandstein', 'Schalstein', 'Schiefer', 'Schwarzpelit', 'Serpentinit', 'Shonkinit', 'Silikat-Perowskit', 'Siltstein', 'Skarn', 'Sonnenbrennerbasalt', 'Sövit', 'Spessartit', 'Spiculit', 'Spilit', 'Steinkohle', 'Steinsalz', 'Steinmeteorit', 'Suevit', 'Syenit', 'Talk-Disthen-Schiefer', 'Tektit', 'Tephrit', 'Teschenit', 'Tachylit', 'Theralith', 'Tholeiit', 'Tonalit', 'Tonschiefer', 'Tonstein', 'Trachyt', 'Travertin', 'Troktolith', 'Trondhjemit', 'Tropfstein', 'Tuffstein', 'Unakit', 'Verit', 'Weißschiefer', 'Websterit', 'Wyomingit']  # noqa


class Stone:
    """
    A stone is described as a collection of vertices (n, 3), aligned to the coordinate axis
    length in x-direction, width in y-direction, height in z-direction
    """
    name = None
    # original data and orientation
    vertices_orig = None
    eigenvalue_orig = None
    eigenvector_orig = None
    order = None  # indexes to height, width, length (ascending) in the original pca
    # transformed vertices and orientation
    vertices = None

    def __init__(self, vertices: np.ndarray, name: str = None):
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
        self.order = np.argsort(self.eigenvalue_orig).tolist()
        r = self.rot_matrix(self.eigenvector_orig, order=self.order)

        self.vertices, self.center = self.transform2origin(vertices, r)

    @staticmethod
    def pca(vert: np.ndarray):
        """
        Calculates the principal component analysis of vertices (n, 3) and returns the eigenvalue and eigenvectors

        :param vert: collection of vertices
        :return: eigenvalues, eigenvectors
        """
        cov_matrix = np.cov(vert.T)
        e_val, e_vec = np.linalg.eig(cov_matrix)

        return e_val, e_vec

    # Calculate the rotation matrix from eigenvectors (basis) to a new basis (coordinate axis)
    @staticmethod
    def rot_matrix(e_vec: np.ndarray, order: List = (2, 1, 0),
                   x=np.array([1, 0, 0]), y=np.array([0, 1, 0]), z=np.array([0, 0, 1])) -> np.ndarray:
        """
        Calculates the rotation matrix from the three eigenvectors to the coordinate axis.
        If an order of the eigenvectors is given (ascending), the rotation is defined,
        that the strongest eigenvector is rotated to the x-axis and so on.
        It solves the linear matrix equation ``ax=b`` with the (ordered) eigenvectors ``(e1, e2, e3)`` as ``a``,
        the coordinate axis ``(x, y, z)`` as ``b``::

           R * e_l = x    |(el, ew, ez) (zero (3,3)) (zero (3,3))|    | a |   |x|
           R * e_w = y -> |(zero (3,3)) (el, ew, ez) (zero (3,3))| *  |...| = |y|
           R * e_z = z    |(zero (3,3)) (zero (3,3)) (el, ew, ez)|    | h |   |z|
           (3, 3) * (3, 1) = (3, 1)  -> (9, 9) * (9, 1) = (9, 1)

           reshape x(9, 1) to R(3, 3)

        :param e_vec: Array of the three (ordered) eigenvectors
        :param order: order of the eigenvectors (ascending). With the default values, no ordering is applied
        :param x: x-axis: length will be transformed to this axis
        :param y: y-axis: width will be transformed to this axis
        :param z: z-axis: height will be transformed to this axis
        :return: Rotation matrix R (3, 3)
        """
        # Rearrange the eigenvectors according to the given order
        # if the default ordering is selected (2, 1, 0) orig == e_vec
        orig = np.array([e_vec[:, order[2]], e_vec[:, order[1]], e_vec[:, order[0]]]).T
        zero = np.zeros((3, 3))

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
        Calculate the center of mass of a given set of vertices (uniform mass distribution)::

           R * v = v' with R = (3, 3), v = (3, n), v' = (3, n)

        :param vert: collection of vertices (n, 3)
        :param r: rotation matrix
        :return: Transformed vertices and the (new) center
        """
        # Apply the rotation: R*v = v'
        vert_rot = np.matmul(r, vert.T).T
        # calculate the center (mean of each coordinates)
        m = np.mean(vert_rot, axis=0)
        # shift the center to the origin (0, 0, 0)
        return vert_rot - m, m-m

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
