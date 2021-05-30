# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 02.03.21, test_sim.py
#

import random
import os

from yade import ymport, qt, polyhedra_utils
from yade.wrapper import Omega, CpmMat, PolyhedraMat, Polyhedra, O
# from yade import Polyhedra
# from yade.polyhedra_utils import polyhedra
# import numpy as np
# import pymesh

# from trockenmauer import Boundary
from trockenmauer.generate_stones import generate_regular_stone
# from trockenmauer.math_utils import Translation

# create a temporary folder for files
folder = 'temp'
if not os.path.isdir(folder):
    os.mkdir(folder)

# place stones within x [0, 2], y [0, 1] -> 2m long, 1m wide
wall_x0, wall_x1 = [0, 2]
wall_y0, wall_y1 = [0, 1]

stones = [generate_regular_stone(0.2, 0.1, 0.1) for _ in range(3)]
# boundary = Boundary()
# boundary_file = f'{folder}/boundary.mesh'
# pymesh.save_mesh(boundary_file, boundary.mesh_solid, ascii=True)
# place the stone random within th wall
# x = random.uniform(wall_x0, wall_x1)
# y = random.uniform(wall_y0, wall_y1)
z = 1
# t = np.array([x, y, z]))
# stone.transform(Translation(np.array([1, .5, 1])))
print('center', stones[0].center)

# Omega is the scene?
o = Omega()

# surface at the bottom
# o.bodies.append(utils.wall(0, axis=2, sense=1, material=m))

# add a concrete material
# mat = CpmMat()
mat = PolyhedraMat()
mat.density = 2600  # kg/m^3
mat.young = 1E6  # Pa
mat.poisson = 20000 / 1E6
mat.frictionAngle = 0.6  # rad
print(mat.dict())  # print all attributes

o.materials.append(mat)
# b = ymport.gmsh(boundary_file, scale=1., material=mat, color=(0, 0, 1), mask=1, fixed=True)
# boundary: plane at z=0
floor = utils.wall(0, axis=2, sense=1, material=mat)
# stones
stone = stones[0]
p = polyhedra_utils.polyhedra(mat, v=stone.mesh.vertices)
p.state.pos = (1, .5, 1)
print(p.dict())
print(Polyhedra.GetCentroid(p.shape))
o.bodies.append(floor)
o.bodies.append(p)

# setup the animation
def checkUnbalanced():
    if utils.unbalancedForce() > 0.0001:
        print('no unalanced forces')
    else:
        # at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
        print("unbalanced forces = %.5f, position %f, %f, %f" % (
        utils.unbalancedForce(), p.state.pos[0], p.state.pos[1], p.state.pos[2]))


"""o.engines = [
    ForceResetter(),
    # Bounds for objects: bo1 -> aabb for the object
    # InsertionSortCollider sorts the bounds and then colides
    InsertionSortCollider([Bo1_Polyhedra_Aabb(), Bo1_Facet_Aabb(), Bo1_Sphere_Aabb()]),  # , Bo1_Wall_Aabb(), Bo1_Facet_Aabb()]),

    InteractionLoop(
        # InteractionGeometry -> Based on 2 types of geometry
        [Ig2_Polyhedra_Polyhedra_PolyhedraGeom()],  # Ig2_Wall_Polyhedra_PolyhedraGeom(), Ig2_Facet_Polyhedra_PolyhedraGeom()],
        # InteractionPhysics -> Based on 2 types of materials
        [Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys()],  # collision "physics"
        # ConstitutiveLaw: they resolve the interaction by computing forces on the interacting bodies (repulsion, attraction, shear forces, …)
        [Law2_PolyhedraGeom_PolyhedraPhys_Volumetric()]  # contact law -- apply forces
    ),
    # GravityEngine(gravity=(0,0,-9.81)),
    NewtonIntegrator(damping=0.5, gravity=(0, 0, -9.81)),
    PyRunner(command='checkUnbalanced()', realPeriod=3, label='checker')
]"""
o.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Polyhedra_Aabb(), Bo1_Wall_Aabb(), Bo1_Facet_Aabb()], verletDist=.001),
    InteractionLoop(
        [Ig2_Wall_Polyhedra_PolyhedraGeom(), Ig2_Polyhedra_Polyhedra_PolyhedraGeom(),
         Ig2_Facet_Polyhedra_PolyhedraGeom()],
        [Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys()],  # collision "physics"
        [Law2_PolyhedraGeom_PolyhedraPhys_Volumetric()]  # contact law -- apply forces
    ),
    # GravityEngine(gravity=(0,0,-9.81)),
    NewtonIntegrator(damping=0.5, gravity=(0, 0, -9.81)),
    PyRunner(command='checkUnbalanced()', realPeriod=3, label='checker')

]

o.dt = 0.00005
qt.Controller()
V = qt.View()

o.saveTmp()
# o.run()

utils.waitIfBatch()