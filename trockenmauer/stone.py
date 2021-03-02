# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, stone.py
#

import random

import numpy as np
import matplotlib.pyplot as plt

from .utils import set_axes_equal
from .math_utils import pca, rot_matrix, transform, transform2origin


NAMES = ['Achondrit', 'Adakit', 'Aleurit', 'Alkaligranit', 'Alnöit', 'Alvikit', 'Amphibolit', 'Anatexit', 'Andesit', 'Anhydrit', 'Anorthosit', 'Anthrazit', 'Aplit', 'Arenit', 'Arkose', 'Augengneis', 'Basalt', 'Basanit', 'Bauxit', 'Beforsit', 'Bentonit', 'Bergalith', 'Bimsstein', 'Biolithe', 'Blätterkohle', 'Blauschiefer', 'Bohnerz', 'Braunkohle', 'Brekzie', 'Buntsandstein', 'Bändererz', 'Buchit', 'Cancalit', 'Charnockit', 'Chert', 'Chloritschiefer', 'Chondrit', 'Cipollino', 'Dachschiefer', 'Dacit', 'Diabas', 'Diamiktit', 'Diatomit', 'Diorit', 'Dolerit', 'Dolomit', 'Dunit', 'Ehrwaldit', 'Eisenmeteorit', 'Eisenoolith', 'Eklogit', 'Enderbit', 'Erbsenstein', 'Essexit', 'Evaporit', 'Fanglomerat', 'Faserkohle', 'Felsit', 'Fenit', 'Fettkohle', 'Feuerstein', 'Fladenlava', 'Flammkohle', 'Fleckschiefer', 'Flint', 'Flysch', 'Foidit', 'Fortunit', 'Foyait', 'Fruchtschiefer', 'Fulgurit', 'Gabbro', 'Garbenschiefer', 'Gauteit', 'Gasflammkohle', 'Gaskohle', 'Gips', 'Glanz(braun)kohle', 'Glaukophanschiefer', 'Glimmerschiefer', 'Gneis', 'Granit', 'Granitporphyr', 'Granodiorit', 'Granophyr', 'Granulit', 'Graptolithenschiefer', 'Grauwacke', 'Griffelschiefer', 'Grünschiefer', 'Hälleflinta', 'Halitit', 'Hartbraunkohle', 'Harzburgit', 'Hawaiit', 'Hornblendit', 'Hornfels', 'Hornstein', 'Ignimbrit', 'Impaktit', 'Itakolumit', 'Jacupirangit', 'Jumillit', 'Kakirit', 'Kalisalze', 'Kalksandstein', 'Kalkstein', 'Kalksilikatfels', 'Kalksinter', 'Kalktuff', 'Kalziolith', 'Kännelkohle', 'Kaolin', 'Karbonatit', 'Karstmarmore', 'Kataklasit', 'Kennelkohle', 'Keratophyr', 'Kersantit', 'Khondalit', 'Kieselerde', 'Kieselgur', 'Kieselschiefer', 'Kieselsinter', 'Kimberlit', 'Kissenlava', 'Klingstein', 'Knochenbrekzie', 'Knotenschiefer', 'Kohle', 'Kohleeisenstein', 'Kohlenkalk', 'Kokardenerz', 'Konglomerat', 'Kontaktschiefer', 'Korallenerz', 'Kreide', 'Kuckersit', 'Lamproit', 'Lamprophyr', 'Lapilli', 'Lapislazuli', 'Larvikit', 'Lava', 'Latit', 'Lehm', 'Leptynit', 'Letten', 'Leucitit', 'Lherzolith', 'Lignit', 'Limburgit', 'Listwänit', 'Liparit', 'Liptobiolith', 'Lockergestein', 'Löss', 'Lutit', 'Lydit', 'Madupit', 'Magerkohle', 'Mafitit', 'Mandelstein', 'Manganknollen', 'Marmor', 'Massenkalk', 'Mattkohle', 'Meimechit', 'Melaphyr', 'Melilithit', 'Mergel', 'Mergelschiefer', 'Mergelstein', 'Meteorit', 'Migmatit', 'Mikrogabbro', 'Mikrogranit', 'Minette (Ganggestein)', 'Minette (Erz)', 'Moldavit', 'Monchiquit', 'Monzonit', 'MORB', 'Mugearit', 'Mylonit', 'Nephelinbasalt', 'Nephelinit', 'Nephelinsyenit', 'Norit', 'Obsidian', 'OIB', 'Ölschiefer', 'Oolith', 'Ophicalcit', 'Ophiolith', 'Ophit', 'Orendit', 'Pallasit', 'Pechstein', 'Pantellerit', 'Pegmatit', 'Perlit', 'Peridotit', 'Phonolith', 'Phyllit', 'Pikrit', 'Pläner', 'Polzenit', 'Porphyr', 'Porphyrit', 'Prasinit', 'Pseudotachylit', 'Pyroxenit', 'Quarzit', 'Quarzolith', 'Quarzporphyr', 'Radiolarit', 'Rapakiwi', 'Raseneisenstein', 'Rauhaugit', 'Rhyolith', 'Rodingit', 'Rogenstein', 'Sagvandit', 'Sannait', 'Sandstein', 'Schalstein', 'Schiefer', 'Schwarzpelit', 'Serpentinit', 'Shonkinit', 'Silikat-Perowskit', 'Siltstein', 'Skarn', 'Sonnenbrennerbasalt', 'Sövit', 'Spessartit', 'Spiculit', 'Spilit', 'Steinkohle', 'Steinsalz', 'Steinmeteorit', 'Suevit', 'Syenit', 'Talk-Disthen-Schiefer', 'Tektit', 'Tephrit', 'Teschenit', 'Tachylit', 'Theralith', 'Tholeiit', 'Tonalit', 'Tonschiefer', 'Tonstein', 'Trachyt', 'Travertin', 'Troktolith', 'Trondhjemit', 'Tropfstein', 'Tuffstein', 'Unakit', 'Verit', 'Weißschiefer', 'Websterit', 'Wyomingit']  # noqa


class Stone:
    """
    A stone is described as a collection of vertices (n, 3), aligned to the coordinate axis
    length in x-direction, width in y-direction, height in z-direction
    """
    name = None
    # original data and orientation
    vertices_orig = None  # (n, 3)
    eigenvalue_orig = None
    eigenvector_orig = None  # (3, 3) !!! vectors as column vectors !!!
    order = None  # indexes to height, width, length (decreasing) in the original pca
    # transformed vertices and orientation
    vertices = None  # (n, 3)

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
        self.order = np.argsort(self.eigenvalue_orig)[::-1]  # decreasing from strongest to weakest
        r = rot_matrix(self.eigenvector_orig, order=self.order)

        self.vertices, self.center = self.transform2origin(vertices, r)

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

    def transform(self, r=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), t=np.array([0, 0, 0])):
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

    def add_plot_to_ax(self, ax, orig=False, positive_eigenvec=True):

        if orig:
            vert = self.vertices_orig
        else:
            vert = self.vertices
        # Plot the points
        ax.plot3D(vert[:, 0], vert[:, 1], vert[:, 2], 'g.')

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
