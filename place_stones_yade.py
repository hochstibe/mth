# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 02.03.21, place_stones_yade.py
#

from yade import qt, polyhedra_utils, utils
from yade.wrapper import (
    O, PolyhedraMat, ForceResetter, InsertionSortCollider, Bo1_Polyhedra_Aabb, Bo1_Wall_Aabb, Bo1_Facet_Aabb,
    InteractionLoop, Ig2_Wall_Polyhedra_PolyhedraGeom, Ig2_Polyhedra_Polyhedra_PolyhedraGeom,
    Ig2_Facet_Polyhedra_PolyhedraGeom, Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys,
    Law2_PolyhedraGeom_PolyhedraPhys_Volumetric, NewtonIntegrator, PyRunner
)
import numpy as np

from trockenmauer.generate_stones import generate_regular_stone

STONES = 3

# Generate Llist of stones for future placements
stones = [generate_regular_stone(0.2, 0.1, 0.1) for _ in range(STONES+1)]
# print('center', stones[0].center)

# add a concrete material
# mat = CpmMat()
mat = PolyhedraMat()
mat.density = 2600  # kg/m^3
# mat.young = 1E6  # Pa
# mat.poisson = 20000 / 1E6
# mat.frictionAngle = 0.6  # rad
# print(mat.dict())  # print all attributes

O.materials.append(mat)
# b = ymport.gmsh(boundary_file, scale=1., material=mat, color=(0, 0, 1), mask=1, fixed=True)
# boundary: plane at z=0
floor = utils.wall(0, axis=2, sense=1, material=mat)
# stones
stones = [polyhedra_utils.polyhedra(mat, v=stone.mesh.vertices) for stone in stones]
stone = stones.pop(0)
stone.state.pos = (1, .5, 1)
# Add the bodies to the scene
O.bodies.append(floor)
O.bodies.append(stone)

PLACED_STONES = 0
# setup the animation
def checkUnbalanced():
    global PLACED_STONES  # , wall, stones
    force = utils.unbalancedForce()
    if np.isnan(force) or force > 0.002:
        # at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
        # print("unbalanced forces = %.5f, position %f, %f, %f" % (
        # utils.unbalancedForce(), p.state.pos[0], p.state.pos[1], p.state.pos[2]))
        print('unbalanced forces =', force)
    else:
        print('no unalanced forces')
        if PLACED_STONES < STONES:
            print('--> add a stone')
            # last stone fixed -> not possible via attributes: remove and create a new object?
            # Pause the engine
            # O.pause()
            # find a suitable placement
            s = stones[PLACED_STONES]
            print('--> find a suitable placement')
            # res = solve_placement(wall, s, 3, 3, validator)
            # print(f'--> {res.position}, {res.value}')
            # p = polyhedra_utils.polyhedra(mat, v=s.mesh.vertices)
            s.state.pos = (1, .5, .1)
            O.bodies.append(s)
            PLACED_STONES += 1
            # O.run()
        else:
            print('all stones placed')


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
    # GravityEngine(gravity=(0, 0, -9.81)),
    NewtonIntegrator(damping=.9, gravity=(0, 0, -9.81)),  # damping=.5
    # Check for unbalanced forces every second
    PyRunner(command='checkUnbalanced()', realPeriod=1, label='checker')

]

O.dt = 0.0001
qt.Controller()
V = qt.View()

# O.saveTmp()  # Save the simulation, that it can used later
# o.run()

# utils.waitIfBatch()  # wait, batch file execution
