# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 25.02.2021, utils.py
#

from typing import TYPE_CHECKING, Union

from .stone import Geometry, Stone, Boundary, Intersection

if TYPE_CHECKING:
    from pymesh import Mesh


def load_from_pymesh(geom_type: str, mesh: 'Mesh', name: str = None
                     ) -> Union['Geometry', 'Stone', 'Boundary']:
    """
    Load a geometry from a pymesh object

    :param geom_type: 'geometry', 'stone', 'boundary', 'intersection'
    :param mesh: pymesh object with vertices and faces
    :param name: optional name of the stone
    :return: Stone
    """

    if geom_type.lower() == 'geometry':
        return Geometry(mesh, name)
    elif geom_type.lower() == 'stone':
        return Stone(mesh.vertices, mesh.faces, name)
    elif geom_type.lower() == 'boundary':
        return Boundary(mesh.vertices, mesh.faces, name)
    elif geom_type.lower() == 'intersection':
        return Intersection(mesh, name)
    else:
        raise ValueError("Type must be one of 'geometry', 'stone', 'boundary', 'intersection'")
