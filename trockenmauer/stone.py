# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, stone.py
#

import random
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pymesh

from .math_utils import order_clockwise, pca, rot_matrix, RotationTranslation, Rotation, Transformation, tetra_volume

if TYPE_CHECKING:
    from .firefly import Iteration, Firefly


NAMES = ['Achondrit', 'Adakit', 'Aleurit', 'Alkaligranit', 'Alnöit', 'Alvikit', 'Amphibolit', 'Anatexit', 'Andesit', 'Anhydrit', 'Anorthosit', 'Anthrazit', 'Aplit', 'Arenit', 'Arkose', 'Augengneis', 'Basalt', 'Basanit', 'Bauxit', 'Beforsit', 'Bentonit', 'Bergalith', 'Bimsstein', 'Biolithe', 'Blätterkohle', 'Blauschiefer', 'Bohnerz', 'Braunkohle', 'Brekzie', 'Buntsandstein', 'Bändererz', 'Buchit', 'Cancalit', 'Charnockit', 'Chert', 'Chloritschiefer', 'Chondrit', 'Cipollino', 'Dachschiefer', 'Dacit', 'Diabas', 'Diamiktit', 'Diatomit', 'Diorit', 'Dolerit', 'Dolomit', 'Dunit', 'Ehrwaldit', 'Eisenmeteorit', 'Eisenoolith', 'Eklogit', 'Enderbit', 'Erbsenstein', 'Essexit', 'Evaporit', 'Fanglomerat', 'Faserkohle', 'Felsit', 'Fenit', 'Fettkohle', 'Feuerstein', 'Fladenlava', 'Flammkohle', 'Fleckschiefer', 'Flint', 'Flysch', 'Foidit', 'Fortunit', 'Foyait', 'Fruchtschiefer', 'Fulgurit', 'Gabbro', 'Garbenschiefer', 'Gauteit', 'Gasflammkohle', 'Gaskohle', 'Gips', 'Glanz(braun)kohle', 'Glaukophanschiefer', 'Glimmerschiefer', 'Gneis', 'Granit', 'Granitporphyr', 'Granodiorit', 'Granophyr', 'Granulit', 'Graptolithenschiefer', 'Grauwacke', 'Griffelschiefer', 'Grünschiefer', 'Hälleflinta', 'Halitit', 'Hartbraunkohle', 'Harzburgit', 'Hawaiit', 'Hornblendit', 'Hornfels', 'Hornstein', 'Ignimbrit', 'Impaktit', 'Itakolumit', 'Jacupirangit', 'Jumillit', 'Kakirit', 'Kalisalze', 'Kalksandstein', 'Kalkstein', 'Kalksilikatfels', 'Kalksinter', 'Kalktuff', 'Kalziolith', 'Kännelkohle', 'Kaolin', 'Karbonatit', 'Karstmarmore', 'Kataklasit', 'Kennelkohle', 'Keratophyr', 'Kersantit', 'Khondalit', 'Kieselerde', 'Kieselgur', 'Kieselschiefer', 'Kieselsinter', 'Kimberlit', 'Kissenlava', 'Klingstein', 'Knochenbrekzie', 'Knotenschiefer', 'Kohle', 'Kohleeisenstein', 'Kohlenkalk', 'Kokardenerz', 'Konglomerat', 'Kontaktschiefer', 'Korallenerz', 'Kreide', 'Kuckersit', 'Lamproit', 'Lamprophyr', 'Lapilli', 'Lapislazuli', 'Larvikit', 'Lava', 'Latit', 'Lehm', 'Leptynit', 'Letten', 'Leucitit', 'Lherzolith', 'Lignit', 'Limburgit', 'Listwänit', 'Liparit', 'Liptobiolith', 'Lockergestein', 'Löss', 'Lutit', 'Lydit', 'Madupit', 'Magerkohle', 'Mafitit', 'Mandelstein', 'Manganknollen', 'Marmor', 'Massenkalk', 'Mattkohle', 'Meimechit', 'Melaphyr', 'Melilithit', 'Mergel', 'Mergelschiefer', 'Mergelstein', 'Meteorit', 'Migmatit', 'Mikrogabbro', 'Mikrogranit', 'Minette (Ganggestein)', 'Minette (Erz)', 'Moldavit', 'Monchiquit', 'Monzonit', 'MORB', 'Mugearit', 'Mylonit', 'Nephelinbasalt', 'Nephelinit', 'Nephelinsyenit', 'Norit', 'Obsidian', 'OIB', 'Ölschiefer', 'Oolith', 'Ophicalcit', 'Ophiolith', 'Ophit', 'Orendit', 'Pallasit', 'Pechstein', 'Pantellerit', 'Pegmatit', 'Perlit', 'Peridotit', 'Phonolith', 'Phyllit', 'Pikrit', 'Pläner', 'Polzenit', 'Porphyr', 'Porphyrit', 'Prasinit', 'Pseudotachylit', 'Pyroxenit', 'Quarzit', 'Quarzolith', 'Quarzporphyr', 'Radiolarit', 'Rapakiwi', 'Raseneisenstein', 'Rauhaugit', 'Rhyolith', 'Rodingit', 'Rogenstein', 'Sagvandit', 'Sannait', 'Sandstein', 'Schalstein', 'Schiefer', 'Schwarzpelit', 'Serpentinit', 'Shonkinit', 'Silikat-Perowskit', 'Siltstein', 'Skarn', 'Sonnenbrennerbasalt', 'Sövit', 'Spessartit', 'Spiculit', 'Spilit', 'Steinkohle', 'Steinsalz', 'Steinmeteorit', 'Suevit', 'Syenit', 'Talk-Disthen-Schiefer', 'Tektit', 'Tephrit', 'Teschenit', 'Tachylit', 'Theralith', 'Tholeiit', 'Tonalit', 'Tonschiefer', 'Tonstein', 'Trachyt', 'Travertin', 'Troktolith', 'Trondhjemit', 'Tropfstein', 'Tuffstein', 'Unakit', 'Verit', 'Weißschiefer', 'Websterit', 'Wyomingit']  # noqa


class Geometry:
    """
    Generic class for a geometrical object
    """
    name: str = None
    _mesh: 'pymesh.Mesh' = None
    triangles_values: np.ndarray = None  # triangle coordinates
    _aabb_limits: np.ndarray = None  # np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    _aabb_volume: float = None
    _aabb_area: float = None
    # length / width / height are independent of the orientation and rotation of the geometry
    _length: float = None
    _width: float = None
    _height: float = None

    # visualization
    color: str = 'gray'
    alpha: float = .5

    def __init__(self, mesh: 'pymesh.Mesh' = None, name: str = None):
        self.name = name
        if mesh:
            self.mesh = mesh

    def _calc_triangle_values(self):
        return np.array([[self.mesh.vertices[j] for j in t_ind] for t_ind in self.mesh.faces])

    def _calc_aabb_limits(self):
        return np.array([np.min(self.mesh.vertices, axis=0), np.max(self.mesh.vertices, axis=0)])

    def _update_mesh_properties(self):
        self.triangles_values = self._calc_triangle_values()
        self.aabb_limits = self._calc_aabb_limits()

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: 'pymesh.Mesh'):
        self._mesh = mesh
        self._update_mesh_properties()

    @property
    def aabb_limits(self):
        if not np.any(self._aabb_limits) and self.mesh:
            # no aabb yet but there is a mesh -> use the setter to calculate volume, area and l/w/h
            self.aabb_limits = self._calc_aabb_limits()
        return self._aabb_limits

    @aabb_limits.setter
    def aabb_limits(self, limits: 'np.ndarray'):
        # what if there is a mesh and aabb_limits --> have to be the same
        self._aabb_limits = limits
        self._aabb_volume = float(np.prod(limits[1] - limits[0]))
        self._aabb_area = float(np.prod(limits[1][:2] - limits[0][:2]))
        diff = limits[1] - limits[0]
        diff.sort()
        self._height, self._width, self._length = diff

    @property
    def aabb_volume(self):
        # calculated the first time accessed, no setter possible
        if not np.any(self._aabb_volume) and np.any(self._aabb_limits):
            # no vol yet but there is a aabb
            self._aabb_volume = float(np.prod(self._aabb_limits[1] - self._aabb_limits[0]))
        return self._aabb_volume

    @property
    def aabb_area(self):
        # calculated the first time accessed, no setter possible
        if not np.any(self._aabb_area) and np.any(self._aabb_limits):
            # no area yet but there is a aabb
            self._aabb_area = float(np.prod(self._aabb_limits[1][:2] - self._aabb_limits[0][:2]))
        return self._aabb_area

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def aabb_overlap(self, other_limits: 'np.ndarray') -> Optional['np.ndarray']:
        """
        calculates the overlapping region. if no overlap, it returns None

        :param other_limits: np.arr
        :return: limits
        """

        # minimum of both maxima; maximum of both minima
        overlap_min = np.max(np.array([self.aabb_limits[0], other_limits[0]]), axis=0)
        overlap_max = np.min(np.array([self.aabb_limits[1], other_limits[1]]), axis=0)

        # difference is positive, if there is an overlap
        diff = overlap_max - overlap_min

        if np.any(diff <= 0):  # todo: ?
            # adjacent bb, the tree returns an overlap
            # print('No overlap, diff:', diff)
            return None

        return np.array([overlap_min, overlap_max])

    def add_shape_to_ax(self, ax):
        # Plot the points (with Poly3DCollection, the extents of the plot is not calculate
        ax.plot3D(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.vertices[:, 2],
                  color=self.color, alpha=self.alpha, marker='.', markersize=.25)
        # triangulation
        col = Poly3DCollection(self.triangles_values, linewidths=.25, edgecolors='black', alpha=self.alpha)
        col.set_facecolor(self.color)
        ax.add_collection3d(col)

    def __repr__(self):
        return f'<Geometry(name={self.name})>'


class Intersection(Geometry):
    """
    Intersection of geometries, init with a mesh or with bb_limits
    """
    mesh_volume: float = None

    def __init__(self, mesh: 'pymesh.Mesh' = None, bb_limits: 'np.ndarray' = None, name: str = None):
        # if there is a mesh, __init__ adds the mesh, calculates triangle values and the aabb
        super().__init__(mesh, name)

        if np.any(bb_limits):
            self.aabb_limits = bb_limits

        if self.mesh and np.any(self.mesh.voxels):
            self.mesh_volume = tetra_volume(mesh.vertices.T, mesh.voxels)


class Boundary(Geometry):
    """
    The boundary of the wall (same width at top as at the bottom)
    a, b, c, d: bottom 4 vertices (counterclockwise)
    e, f, g h: top 4 vertices (clockwise
    """
    x: float = 0  # length
    y: float = 0  # width
    z: float = 0  # height
    batter: float = 0  # 'Anzug' fraction of the height, the taler, the narrower
    # self.mesh is only for visualizations (5 triangulated planes, no closed mesh)
    # Solid meshes can be used for boolean operations
    mesh_solid: 'pymesh.Mesh' = None  # with the bottom boundary
    mesh_solid_sides: 'pymesh.Mesh' = None  # only boundaries at the sides

    def __init__(self, x=2., y=0.5, z=1, batter=0, name='Boundary'):
        """

        :param x: Length [m]
        :param y: Width [m]
        :param z: Height [m]
        """
        super().__init__(name=name)
        self.color = 'grey'
        self.alpha = .1
        self.x = x
        self.y = y
        self.z = z
        self.batter = batter
        self.volume = x * (y + y - 2*y*batter)/2 *z

        ba = self.batter * self.z

        # bounding vertices
        # lower
        a, b, c, d = [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0]
        # upper
        e, f, g, h = [0, ba, z], [x, ba, z], [x, y-ba, z], [0, y-ba, z]
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
        self.mesh = pymesh.form_mesh(vertices[:8], triangles_index[:10])

        # the solid mesh can be used for boolean operations (intersection) -> not used at the moment
        self.mesh_solid = pymesh.form_mesh(vertices, triangles_index)

        # solid mesh without the bottom plane for calculating the distance to the sides
        triangles_index = np.array([
            [0, 5, 1], [0, 4, 5],  # front
            [2, 7, 3], [2, 6, 7],  # back
            [3, 4, 0], [3, 7, 4],  # left
            [1, 6, 2], [1, 5, 6],  # right
            # outer for creating a solid
            [0, 1, 8], [1, 9, 8], [1, 2, 9], [2, 10, 9],  # bottom
            [2, 3, 10], [3, 11, 10], [3, 0, 11], [0, 8, 11],  # bottom
            [8, 9, 13], [8, 13, 12],  # front
            [10, 11, 15], [10, 15, 14],  # back
            [11, 8, 12], [11, 12, 15],  # left
            [9, 10, 14], [9, 14, 13],  # right
            [12, 13, 5], [12, 5, 4], [13, 14, 6], [13, 6, 5],  # top
            [14, 15, 7], [14, 7, 6], [15, 12, 4], [15, 4, 7],  # top
        ])
        self.mesh_solid_sides = pymesh.form_mesh(vertices, triangles_index)

    def __repr__(self):
        return f'<Boundary(vertices=array([{(self.mesh.vertices[:2])}, ...])>'


class Stone(Geometry):
    """
    A stone is described as a collection of vertices (n, 3), initially aligned to the
    coordinate axis length in x-direction, width in y-direction, height in z-direction
    """
    # geometrical properties -> ideally only getter, no setter
    eigenvalue: np.ndarray = None
    eigenvector: np.ndarray = None  # (3, 3) !!! vectors as column vectors !!!
    center: np.ndarray = None
    current_rotation: np.ndarray = np.eye(3)  # center + rotation --> transformation to original position possible

    # faces of the stone and their properties
    bottom: np.ndarray = None  # vertices of the bottom face
    bottom_center: np.ndarray = None  # center of the bottom face
    bottom_n: np.ndarray = None  # normal of the bottom face
    top: np.ndarray = None  # vertices of the top face
    top_center: np.ndarray = None  # center of the top face
    top_n: np.ndarray = None  # normal of the top face

    sides: np.ndarray = None  # list of the sides of the stone (indices to the mesh)
    sides_center: np.ndarray = None  # center point (mean) of each side
    # sides_n: List[np.ndarray] = None

    # History of the firefly positions
    position_history: List['Iteration'] = ()
    best_firefly: 'Firefly' = None

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
        self.color = 'green'

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

            self.bottom = np.array([0, 1, 2, 3])
            self.top = np.array([4, 5, 6, 7])
            self.sides = np.array([[0, 1, 4, 5],  # front
                                   [2, 3, 6, 7],  # back
                                   [0, 3, 4, 7],  # left
                                   [1, 2, 5, 6]])  # right

            # initialize triangles
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
            # The triangles are already defined
            triangles_index = triangles_index

            # set the lower half of the vertices as the bottom vertices
            ind = np.argsort(vertices[:, 2])
            half = int(len(vertices) / 2)
            self.bottom = ind[:half]
            self.top = ind[half:]

            # Todo: sides for arbitrary stones

        self.mesh = pymesh.form_mesh(vertices, triangles_index)

        # calculate the volume
        tetgen = pymesh.tetgen()
        tetgen.max_tet_volume = 2
        tetgen.verbosity = 0
        tetgen.split_boundary = False
        tetgen.points = self.mesh.vertices
        tetgen.triangles = self.mesh.faces
        tetgen.run()
        tetra = tetgen.mesh

        self.mesh_volume = tetra_volume(tetra.vertices.T, tetra.voxels)

    def _update_mesh_properties(self):
        """
        Get important faces and their normals.
        Simplification: a face is a plane for rectangular stones

        :return:
        """
        # triangles + aabb from class Geometry
        super()._update_mesh_properties()

        self.center = self.mesh.vertices.mean(axis=0)
        self.eigenvalue, self.eigenvector = self.pca()

        bottom = self.mesh.vertices[self.bottom]
        self.bottom_center = bottom.mean(axis=0)
        # Normal is a crude approximation (true, if the bottom points lie in a plane)
        self.bottom_n = np.cross(bottom[0] - bottom[1], bottom[2] - bottom[0])

        top = self.mesh.vertices[self.top]
        self.top_center = top.mean(axis=0)
        self.top_n = np.cross(top[1] - top[0], top[2] - top[0])

        # self.height = np.linalg.norm(self.top_center - self.bottom_center)

        if np.any(self.sides):
            self.sides_center = np.array([self.mesh.vertices[s].mean(axis=0) for s in self.sides])

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
        if isinstance(transformation, (Rotation, RotationTranslation)):
            self.current_rotation = transformation.rot_mat @ self.current_rotation
        # c_1 = self.bottom_center
        v = transformation.transform(self.mesh.vertices.T)
        # Rewrite mesh (not possible to update?)
        self.mesh = pymesh.form_mesh(v.T, self.mesh.faces)
        # self.mesh.vertices = v.T
        self._update_mesh_properties()
        # print(self.name, 'from',  np.round(c_1, 5), 'to', np.round(self.bottom_center, 5))

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
                x, y, z = np.abs(vec)
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

    def __repr__(self):
        return f'<Stone(name={self.name}, vertices={len(self.mesh.vertices)})>'
