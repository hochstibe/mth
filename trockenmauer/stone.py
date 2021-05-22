# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, stone.py
#

import random
from typing import List

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .math_utils import order_clockwise, pca, rot_matrix, RotationTranslation, Transformation

from pymesh import Mesh, form_mesh, merge_meshes

NAMES = ['Achondrit', 'Adakit', 'Aleurit', 'Alkaligranit', 'Alnöit', 'Alvikit', 'Amphibolit', 'Anatexit', 'Andesit', 'Anhydrit', 'Anorthosit', 'Anthrazit', 'Aplit', 'Arenit', 'Arkose', 'Augengneis', 'Basalt', 'Basanit', 'Bauxit', 'Beforsit', 'Bentonit', 'Bergalith', 'Bimsstein', 'Biolithe', 'Blätterkohle', 'Blauschiefer', 'Bohnerz', 'Braunkohle', 'Brekzie', 'Buntsandstein', 'Bändererz', 'Buchit', 'Cancalit', 'Charnockit', 'Chert', 'Chloritschiefer', 'Chondrit', 'Cipollino', 'Dachschiefer', 'Dacit', 'Diabas', 'Diamiktit', 'Diatomit', 'Diorit', 'Dolerit', 'Dolomit', 'Dunit', 'Ehrwaldit', 'Eisenmeteorit', 'Eisenoolith', 'Eklogit', 'Enderbit', 'Erbsenstein', 'Essexit', 'Evaporit', 'Fanglomerat', 'Faserkohle', 'Felsit', 'Fenit', 'Fettkohle', 'Feuerstein', 'Fladenlava', 'Flammkohle', 'Fleckschiefer', 'Flint', 'Flysch', 'Foidit', 'Fortunit', 'Foyait', 'Fruchtschiefer', 'Fulgurit', 'Gabbro', 'Garbenschiefer', 'Gauteit', 'Gasflammkohle', 'Gaskohle', 'Gips', 'Glanz(braun)kohle', 'Glaukophanschiefer', 'Glimmerschiefer', 'Gneis', 'Granit', 'Granitporphyr', 'Granodiorit', 'Granophyr', 'Granulit', 'Graptolithenschiefer', 'Grauwacke', 'Griffelschiefer', 'Grünschiefer', 'Hälleflinta', 'Halitit', 'Hartbraunkohle', 'Harzburgit', 'Hawaiit', 'Hornblendit', 'Hornfels', 'Hornstein', 'Ignimbrit', 'Impaktit', 'Itakolumit', 'Jacupirangit', 'Jumillit', 'Kakirit', 'Kalisalze', 'Kalksandstein', 'Kalkstein', 'Kalksilikatfels', 'Kalksinter', 'Kalktuff', 'Kalziolith', 'Kännelkohle', 'Kaolin', 'Karbonatit', 'Karstmarmore', 'Kataklasit', 'Kennelkohle', 'Keratophyr', 'Kersantit', 'Khondalit', 'Kieselerde', 'Kieselgur', 'Kieselschiefer', 'Kieselsinter', 'Kimberlit', 'Kissenlava', 'Klingstein', 'Knochenbrekzie', 'Knotenschiefer', 'Kohle', 'Kohleeisenstein', 'Kohlenkalk', 'Kokardenerz', 'Konglomerat', 'Kontaktschiefer', 'Korallenerz', 'Kreide', 'Kuckersit', 'Lamproit', 'Lamprophyr', 'Lapilli', 'Lapislazuli', 'Larvikit', 'Lava', 'Latit', 'Lehm', 'Leptynit', 'Letten', 'Leucitit', 'Lherzolith', 'Lignit', 'Limburgit', 'Listwänit', 'Liparit', 'Liptobiolith', 'Lockergestein', 'Löss', 'Lutit', 'Lydit', 'Madupit', 'Magerkohle', 'Mafitit', 'Mandelstein', 'Manganknollen', 'Marmor', 'Massenkalk', 'Mattkohle', 'Meimechit', 'Melaphyr', 'Melilithit', 'Mergel', 'Mergelschiefer', 'Mergelstein', 'Meteorit', 'Migmatit', 'Mikrogabbro', 'Mikrogranit', 'Minette (Ganggestein)', 'Minette (Erz)', 'Moldavit', 'Monchiquit', 'Monzonit', 'MORB', 'Mugearit', 'Mylonit', 'Nephelinbasalt', 'Nephelinit', 'Nephelinsyenit', 'Norit', 'Obsidian', 'OIB', 'Ölschiefer', 'Oolith', 'Ophicalcit', 'Ophiolith', 'Ophit', 'Orendit', 'Pallasit', 'Pechstein', 'Pantellerit', 'Pegmatit', 'Perlit', 'Peridotit', 'Phonolith', 'Phyllit', 'Pikrit', 'Pläner', 'Polzenit', 'Porphyr', 'Porphyrit', 'Prasinit', 'Pseudotachylit', 'Pyroxenit', 'Quarzit', 'Quarzolith', 'Quarzporphyr', 'Radiolarit', 'Rapakiwi', 'Raseneisenstein', 'Rauhaugit', 'Rhyolith', 'Rodingit', 'Rogenstein', 'Sagvandit', 'Sannait', 'Sandstein', 'Schalstein', 'Schiefer', 'Schwarzpelit', 'Serpentinit', 'Shonkinit', 'Silikat-Perowskit', 'Siltstein', 'Skarn', 'Sonnenbrennerbasalt', 'Sövit', 'Spessartit', 'Spiculit', 'Spilit', 'Steinkohle', 'Steinsalz', 'Steinmeteorit', 'Suevit', 'Syenit', 'Talk-Disthen-Schiefer', 'Tektit', 'Tephrit', 'Teschenit', 'Tachylit', 'Theralith', 'Tholeiit', 'Tonalit', 'Tonschiefer', 'Tonstein', 'Trachyt', 'Travertin', 'Troktolith', 'Trondhjemit', 'Tropfstein', 'Tuffstein', 'Unakit', 'Verit', 'Weißschiefer', 'Websterit', 'Wyomingit']  # noqa


class Geometry:
    """
    Generic class for a geometrical object
    """
    name: str = None
    mesh: 'Mesh' = None
    triangles_values: np.ndarray = None  # triangle coordinates

    def __init__(self, mesh: 'Mesh' = None, name: str = None):
        self.name = name
        if mesh:
            self.mesh = mesh
            self.calc_triangle_values()

    def calc_triangle_values(self):
        self.triangles_values = [[self.mesh.vertices[j] for j in t_ind] for t_ind in self.mesh.faces]

    def add_shape_to_ax(self, ax, color='red'):
        # Plot the points (with Poly3DCollection, the extents of the plot is not calculate
        ax.plot3D(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.vertices[:, 2],
                  color=color, marker='.', markersize=1)
        # triangulation
        col = Poly3DCollection(self.triangles_values, linewidths=1, edgecolors=color, alpha=.1)
        col.set_facecolor(color)
        ax.add_collection3d(col)

    def __repr__(self):
        return f'<Geometry(name={self.name})>'


class Wall:
    """
    A wall consists of the placed stones and the boundary.
    """

    def __init__(self, boundary: 'Boundary', stones: List['Stone'] = None, mesh: 'Mesh' = None):
        self.boundary = boundary
        self.stones = stones
        self.mesh = mesh

        if not self.stones:
            self.stones = []
            # self.mesh = form_mesh(np.array([0, 0, 0]), np.array([]))

    def add_stone(self, stone: 'Stone'):
        # Add the stone to the mesh
        if not self.mesh:
            self.mesh = stone.mesh
        else:
            self.mesh = merge_meshes([self.mesh, stone.mesh])

        # Add the stone to the list to keep track of the individual stones
        self.stones.append(stone)

    def __repr__(self):
        return f'<Wall(boundary={self.boundary}, stones={self.stones})>'


class Boundary(Geometry):
    """
    The boundary of the wall (same width at top as at the bottom)
    a, b, c, d: bottom 4 vertices (counterclockwise)
    e, f, g h: top 4 vertices (clockwise
    """

    def __init__(self, x=2., y=0.5, z=1, name='Boundary'):
        """

        :param x: Length [m]
        :param y: Width [m]
        :param z: Height [m]
        """
        super().__init__(name=name)
        self.x = x
        self.y = y
        self.z = z

        # bounding vertices
        # lower
        a, b, c, d = [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0]
        # upper
        e, f, g, h = [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]
        # vertices for creating a solid
        i, j, k, l = [-1, -1, -1], [x+1, -1, -1], [x+1, y+1, -1], [-1, y+1, -1]
        m, n, o, p = [-1, -1, z], [x+1, -1, z], [x+1, y+1, z], [-1, y+1, z]

        vertices = np.array([a, b, c, d, e, f, g, h,
                             i, j, k, l, m, n, o, p])

        triangles_index = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [0, 5, 1], [0, 4, 5],  # front
            [2, 7, 3], [2, 6, 7],  # back
            [3, 4, 0], [3, 7, 4],  # left
            [1, 6, 2], [1, 5, 6],  # right
            # outer for creating a solid
            [8, 10, 9], [8, 11, 10],  # bottom
            [8, 9, 13], [8, 13, 12],  # front
            [10, 11, 15], [10, 15, 14],  # back
            [11, 8, 12], [11, 12, 15],  # left
            [9, 10, 14], [9, 14, 13],  # right
            [12, 13, 5], [12, 5, 4], [13, 14, 6], [13, 6, 5],  # top
            [14, 15, 7], [14, 7, 6], [15, 12, 4], [15, 4, 7],  # top
        ])

        # the mesh is only with the bounding planes (bottom and sides) for plotting
        self.mesh = form_mesh(vertices[:8], triangles_index[:10])
        # the solid mesh can be used for boolean operations (intersection)
        self.mesh_solid = form_mesh(vertices, triangles_index)
        self.calc_triangle_values()

        self.bottom = np.array([a, b, c, d])
        self.front = np.array([a, b, f, e])
        self.back = np.array([d, c, g, h])
        self.left = np.array([a, e, h, d])
        self.right = np.array([b, f, g, c])

    def add_shape_to_ax(self, ax, color='grey'):
        """
        Adds the boundaries to the plot

        :param ax:
        :param color: color for the plot, e.g. 'g', 'r', 'green'
        :return:
        """
        # Default color is different than from Geometry.add_shape_to_ax()
        # Todo: pass kwargs for setting the Poly3DCollection attributes
        super().add_shape_to_ax(ax, color)

        # # Plot the points (with Poly3DCollection, the extents of the plot is not calculate
        # ax.plot3D(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.vertices[:, 2],
        #           color=color, marker='.', markersize=1)
        # # triangulation
        # col = Poly3DCollection(self.triangles_values, linewidths=1, edgecolors=color, alpha=.1)
        # col.set_facecolor(color)
        # ax.add_collection3d(col)

    def __repr__(self):
        return f'<Boundary(vertices=array([{(self.mesh.vertices[:2])}, ...])>'


class Stone(Geometry):
    """
    A stone is described as a collection of vertices (n, 3), initially aligned to the
    coordinate axis length in x-direction, width in y-direction, height in z-direction
    """
    name: str = None

    # geometry: vertices and triangles
    # vertices: np.ndarray = None  # (n, 3)
    # triangles_index: np.ndarray = None   # indices to the triangle points
    triangles_values: np.ndarray = None  # triangle coordinates

    # geometrical properties
    eigenvalue: np.ndarray = None
    eigenvector: np.ndarray = None  # (3, 3) !!! vectors as column vectors !!!
    center: np.ndarray = None

    # faces of the stone and their properties
    bottom_index: np.ndarray = None  # vertices of the bottom face
    bottom_center: np.ndarray = None  # center of the bottom face
    bottom_n: np.ndarray = None  # normal of the bottom face
    top_index: np.ndarray = None  # vertices of the top face
    top_center: np.ndarray = None  # center of the top face
    top_n: np.ndarray = None  # normal of the top face

    height: float

    def __init__(self, vertices: np.ndarray,
                 triangles_index: np.ndarray = None, name: str = None):
        """
        A stone is initially with its center at [0, 0, 0] and the axis are aligned to the coordinate
        axis: length in x-direction, width in y-direction, height in z-direction

        :param vertices: (n, 3)
        :param triangles_index: if the vertices are already triangulated, list of the triangle points
        :param name: optional name for a stone
        """
        super().__init__()

        if name:
            self.name = name
        else:
            self.name = random.choice(NAMES)

        # Array of shape (n, 3)
        if vertices.shape[1] != 3:
            raise ValueError('Input must be of shape (n, 3)')

        # self.vertices = vertices
        self.eigenvalue, self.eigenvector = self.pca(vertices)

        order = np.argsort(self.eigenvalue)[::-1]  # decreasing from strongest to weakest
        r = rot_matrix(self.eigenvector, order=order)

        # Transform the stone to the origin and align the axis
        center = vertices.mean(axis=0)
        t = RotationTranslation(rotation=r, center=center, translation=-center)
        vertices = t.transform(vertices.T).T

        if not np.any(triangles_index) and len(vertices) == 8:
            # get the triangles by hand for rectangular stones

            # order the vertices clockwise bottom, clockwise top
            # order the points by z-value
            ind = np.argsort(vertices[:, 2])

            # get the 4 lowest and 4 highest points
            bottom = vertices[ind[:4]]
            top = vertices[ind[4:]]
            a, b, c, d = order_clockwise(bottom)
            e, f, g, h = order_clockwise(top)

            # update the vertices
            vertices = np.array([a, b, c, d,  # lower 4 vertices
                                 e, f, g, h])  # upper 4 vertices
            # Todo: Top / bottom only index, not coordinates
            self.bottom = vertices[:4]
            self.top = vertices[4:]

            # initialize triangles
            # self.triangles_values = [[] for _ in range(12)]  # 8 vertices -> 12 triangles
            triangles_index = np.array([
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
            ])

        else:
            triangles_index = triangles_index

            # set the lower half of the vertices as the bottom vertices
            ind = np.argsort(vertices[:, 2])
            half = int(len(vertices) / 2)
            self.bottom = vertices[ind[:half]]
            self.top = vertices[ind[half:]]

        self.mesh = form_mesh(vertices, triangles_index)
        self.update_properties()

    def update_properties(self):
        """
        Get important faces and their normals.
        Simplification: a face is a plane for rectangular stones

        :return:
        """

        self.center = self.mesh.vertices.mean(axis=0)
        self.eigenvalue, self.eigenvector = self.pca()

        self.bottom_center = self.bottom.mean(axis=0)
        # Normal is a crude approximation (true, if the bottom points lie in a plane)
        self.bottom_n = np.cross(self.bottom[0] - self.bottom[1], self.bottom[2] - self.bottom[0])

        self.top_center = self.top.mean(axis=0)
        self.top_n = np.cross(self.top[1] - self.top[0], self.top[2] - self.top[0])

        self.height = self.top_center[2] - self.bottom_center[2]

        # update triangle values
        self.calc_triangle_values()
        # self.triangles_values = [[self.vertices[j] for j in t_ind] for t_ind in self.triangles_index]
        # self.triangles_values = [self.vertices[j] for t_ind in self.triangles_index for j in t_ind]

    def pca(self, vertices: np.ndarray = None):
        """
        Calculates the principal component analysis of vertices (n, 3)
        and returns the eigenvalue and eigenvectors

        :return: eigenvalues, eigenvectors
        """
        # Transpose the vertices (util functions for (3, n))
        if not np.any(vertices):
            vertices = self.mesh.vertices

        return pca(vertices.T)

    def transform(self, transformation: Transformation):
        """
        Transforms the vertices (``self.vertices``)

        :param transformation: Transformation object including the transformation matrix
        :return: Updates the vertices and all properties
        """

        v = transformation.transform(self.mesh.vertices.T)
        # Rewrite mesh (not possible to update?)
        self.mesh = form_mesh(v.T, self.mesh.faces)
        # self.mesh.vertices = v.T
        self.update_properties()

    def add_shape_to_ax(self, ax, color: str = 'green'):
        """
        Adds the shape of the stone (triangles) to the plot

        :param ax: pyplot axis
        :param color: color for the plot, e.g. 'g', 'r', 'green'
        :return: -
        """

        # Plot the points
        ax.plot3D(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.vertices[:, 2],
                  color=color, marker='.', markersize=1)
        # Plot the triangles
        col = Poly3DCollection(self.triangles_values, linewidths=0.4, edgecolors=color, alpha=.2)
        col.set_facecolor(color)
        ax.add_collection3d(col)

        # set_axes_equal(ax)

    def add_labels_to_ax(self, ax, positive_eigenvec=False):
        """
        Adds the labels and eigenvectors to the plot

        :param ax: pyplot axis
        :param positive_eigenvec: use the sign of the eigenvectors (left-hand oriented ev appear right-hand)
        :return: -
        """

        # The stones are already aligned to the coordinate axis and centered in (0, 0, 0)
        # calculating the mean and placing the eigenvectors at the mean is not necessary
        # mean = np.mean(self.mesh.vertices, axis=0)

        # Plot the center
        ax.plot3D(self.center[0], self.center[1], self.center[2], 'r.')

        # eigenvalues, eigenvectors = self.pca()

        # Plot the three axes of the stone
        for val, vec in zip(self.eigenvalue, self.eigenvector):
            v = val
            cx, cy, cz = self.center  # center coordinates

            # components of the eigenvector
            # use positive eigenvectors (e_vec can direct wrong direction (180°))
            if positive_eigenvec:
                x, y, z = np.sign(vec)*vec
            # use the correct direction of eigenvectors
            else:
                x, y, z = vec
            ax.plot3D([cx, cx + v*x], [cy, cy + v*y], [cz, cz + v*z], 'r')

            ax.text(cx + v*x, cy + v*y, cz + v*z, np.round(v, 2), 'x')

        # Add the triangle normal to the plot
        # for tri in self.triangles_values:
        #     n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        #     c = np.mean(tri, axis=0)
        #     print(n, c)
        #     ax.plot3D([c[0], c[0]+n[0]], [c[1], c[1]+n[1]], [c[2], c[2]+n[2]], 'b')

    # def add_plot_to_ax(self, ax, positive_eigenvec=False, color='g'):
    #     self.add_shape_to_ax(ax, color)
    #     self.add_labels_to_ax(ax, positive_eigenvec=positive_eigenvec)
    #     return ax

    # def plot(self):
    #     """
    #     Plot the stone
    #     :return:
    #     """
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     self.add_shape_to_ax(ax)
    #     self.add_labels_to_ax(ax)
    #     set_axes_equal(ax)
    #     plt.show()

    def __repr__(self):
        return f'<Stone(name={self.name}, vertices={len(self.mesh.vertices)})>'
