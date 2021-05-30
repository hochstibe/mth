# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 02.03.21, test_sim.py
#

from yade import qt, polyhedra_utils, utils
from yade.wrapper import (
    O, PolyhedraMat, ForceResetter, InsertionSortCollider, Bo1_Polyhedra_Aabb, Bo1_Wall_Aabb, Bo1_Facet_Aabb,
    InteractionLoop, Ig2_Wall_Polyhedra_PolyhedraGeom, Ig2_Polyhedra_Polyhedra_PolyhedraGeom,
    Ig2_Facet_Polyhedra_PolyhedraGeom, Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys,
    Law2_PolyhedraGeom_PolyhedraPhys_Volumetric, NewtonIntegrator, PyRunner
)

# from trockenmauer import Boundary
from trockenmauer.generate_stones import generate_regular_stone
# from trockenmauer.math_utils import Translation


stones = [generate_regular_stone(0.2, 0.1, 0.1) for _ in range(3)]
# print('center', stones[0].center)

# add a concrete material
# mat = CpmMat()
mat = PolyhedraMat()
mat.density = 2600  # kg/m^3
mat.young = 1E6  # Pa
mat.poisson = 20000 / 1E6
mat.frictionAngle = 0.6  # rad
# print(mat.dict())  # print all attributes

O.materials.append(mat)
# b = ymport.gmsh(boundary_file, scale=1., material=mat, color=(0, 0, 1), mask=1, fixed=True)
# boundary: plane at z=0
floor = utils.wall(0, axis=2, sense=1, material=mat)
# stones
stone = stones[0]
p = polyhedra_utils.polyhedra(mat, v=stone.mesh.vertices)
p.state.pos = (1, .5, 1)
# Add the bodies to the scene
O.bodies.append(floor)
O.bodies.append(p)


# setup the animation
def checkUnbalanced():
    if utils.unbalancedForce() > 0.0001:
        print('no unalanced forces')
    else:
        # at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
        print("unbalanced forces = %.5f, position %f, %f, %f" % (
        utils.unbalancedForce(), p.state.pos[0], p.state.pos[1], p.state.pos[2]))


O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Polyhedra_Aabb(), Bo1_Wall_Aabb(), Bo1_Facet_Aabb()], verletDist=.001),
    InteractionLoop(
        [Ig2_Wall_Polyhedra_PolyhedraGeom(), Ig2_Polyhedra_Polyhedra_PolyhedraGeom(),
         Ig2_Facet_Polyhedra_PolyhedraGeom()],
        [Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys()],  # collision "physics"
        [Law2_PolyhedraGeom_PolyhedraPhys_Volumetric()]  # contact law -- apply forces
    ),
    # Add gravity
    NewtonIntegrator(damping=0.5, gravity=(0, 0, -9.81)),
    # Check for unbalanced forces every second
    PyRunner(command='checkUnbalanced()', realPeriod=1, label='checker')

]

O.dt = 0.00005
qt.Controller()
V = qt.View()

O.saveTmp()
# o.run()

utils.waitIfBatch()
