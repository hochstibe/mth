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
        self.order = np.argsort(self.eigenvalue_orig)[::-1]  # decreasing from strongest to weakest
        r = self.rot_matrix(self.eigenvector_orig, order=self.order)

        self.vertices, self.center = self.transform2origin(vertices, r)

    @staticmethod
    def pca(vert: np.ndarray):
        """
        Calculates the principal component analysis of vertices (n, 3)
        and returns the eigenvalue and eigenvectors

        :param vert: collection of vertices
        :return: eigenvalues, eigenvectors
        """
        cov_matrix = np.cov(vert.T)
        e_val, e_vec = np.linalg.eig(cov_matrix)

        return np.sqrt(e_val), e_vec

    # Calculate the rotation matrix from eigenvectors (basis) to a new basis (coordinate axis)
    @staticmethod
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

    @staticmethod
    def transform2origin(vert, r):
        """
        Calculate the center of mass of a given set of vertices (uniform mass distribution)::

           R * v = v' with R = (3, 3), v = (3, n), v' = (3, n)

        :param vert: collection of vertices (n, 3)
        :param r: rotation matrix
        :return: Transformed vertices and the (new) center
        """
        # center of the vertices
        m = np.mean(vert, axis=0)
        # Apply the rotation to the translated vertices: R*(v-m) = v'
        vert_rot = np.matmul(r, (vert - m).T).T
        return vert_rot, np.zeros(3)

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
