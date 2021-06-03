# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 31.05.21, test_sim_2.py
#


import random

from yade import qt, polyhedra_utils, utils
from yade.wrapper import (
    O, PolyhedraMat, ForceResetter, InsertionSortCollider, Bo1_Polyhedra_Aabb, Bo1_Wall_Aabb, Bo1_Facet_Aabb,
    InteractionLoop, Ig2_Wall_Polyhedra_PolyhedraGeom, Ig2_Polyhedra_Polyhedra_PolyhedraGeom,
    Ig2_Facet_Polyhedra_PolyhedraGeom, Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys,
    Law2_PolyhedraGeom_PolyhedraPhys_Volumetric, NewtonIntegrator, PyRunner
)
import numpy as np

from trockenmauer import Wall, Boundary, Validator
from trockenmauer.placement import solve_placement
from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.math_utils import Rotation, RotationTranslation, RZ_90

STONES = 5

# Initialize wall, stones, ...
boundary = Boundary(x=1, y=.5, z=1, batter=.1)
wall = Wall(boundary)
validator = Validator(intersection_boundary=True, intersection_stones=True,
                      distance2boundary=True, volume_below_stone=True,
                      distance2closest_stone=True
                      )
# Generate and place the first stone manually
rz90 = Rotation(rotation=RZ_90, center=np.zeros(3))
stone = generate_regular_stone(.25, 0.15, 0.1, edge_noise=0.5, name=str(-1))
t = np.array([stone.aabb_limits[1][1],
              stone.aabb_limits[1][0] + stone.aabb_limits[1][2] * boundary.batter,
              -stone.bottom_center[2]])
stone.transform(RotationTranslation(rotation=RZ_90, center=np.zeros(3), translation=t))

# Generate list of stones for future placements
stones = [generate_regular_stone(0.2, 0.1, 0.1) for _ in range(STONES)]
for st in stones:
    rotation = random.choice([True, False])
    if rotation:
        st.transform(rz90)
# print('center', stones[0].center)

# add a concrete material
# mat = CpmMat()
mat = PolyhedraMat()
mat.density = 2600  # kg/m^3
# mat.young = 1E6  # Pa
# mat.poisson = 20000 / 1E6  -> stiffness --> increase -> less bouncy
# mat.frictionAngle = 0.6  # rad
# print(mat.dict())  # print all attributes

O.materials.append(mat)
# b = ymport.gmsh(boundary_file, scale=1., material=mat, color=(0, 0, 1), mask=1, fixed=True)
# boundary: plane at z=0
floor = utils.wall(0, axis=2, sense=1, material=mat, color=(.5, .5, .5))
# stones
poly = polyhedra_utils.polyhedra(mat, v=stone.mesh.vertices)
poly.state.pos = stone.center
# Add the bodies to the scene
O.bodies.append(floor)
# Place a boundary -> only for fixing the extents
O.bodies.append(box(center=[-.1, -.1, -.1], extents=[.001, .001, .001], color=[0, 0, 0], fixed=True))
O.bodies.append(box(center=[boundary.x+.1, boundary.y+.1, -.1],
                    extents=[.001, .001, .001], color=[0, 0, 0], fixed=True))
O.bodies.append(poly)
# stones = [polyhedra_utils.polyhedra(mat, v=stone.mesh.vertices) for stone in stones]


# print saved
print('saved simulations', O.lsTmp())

PLACED_STONES = 0
COUNTER = 0
last_id = None
stable_id = str(0)
O.saveTmp(mark=stable_id)
# setup the animation
def checkUnbalanced():
    global PLACED_STONES, wall, stones, COUNTER, last_id, stable_id
    force = utils.unbalancedForce()
    if np.isnan(force) or force > 0.002:
        COUNTER += 1
        # at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
        # print("unbalanced forces = %.5f, position %f, %f, %f" % (
        # utils.unbalancedForce(), p.state.pos[0], p.state.pos[1], p.state.pos[2]))
        print('unbalanced forces =', force)
        if COUNTER >= 5:
            # invalid placement
            # print('!!! invalid placement -> remove stone')
            # O.bodies.erase(last_id)
            print('!!! invalid placement -> load last stable scene', stable_id, O.lsTmp())
            # O.pause()
            O.loadTmp(mark=stable_id)
            # print(O.loadTmp(mark=stable_id))
            # O.reload()
            # O.reset()
            # O.resetCurrentScene()
            # O.run()
            PLACED_STONES += 1
    else:
        print('no unalanced forces')
        print(O.bodies[-1].state.pos, O.bodies[-1].state.ori, O.bodies[-1].state.se3)
        stable_id = str(int(stable_id) + 1)
        O.saveTmp(mark=stable_id)
        COUNTER = 0
        if PLACED_STONES < STONES:
            print('--> add a stone')
            # last stone fixed -> not possible via attributes: remove and create a new object?
            # Pause the engine
            # O.pause()
            # find a suitable placement
            s = stones[PLACED_STONES]
            print('--> find a suitable placement')
            res = solve_placement(wall, s, 3, 3, validator)
            print(f'--> {res.position}, {res.value}')
            p = polyhedra_utils.polyhedra(mat, v=s.mesh.vertices)
            p.state.pos = s.center  # (1, .5, 1)
            last_id = O.bodies.append(p)
            PLACED_STONES += 1
            # O.run()
        else:
            print('all stones placed')
            O.pause()


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

O.dt = 0.0001  # Todo: O.dt=.5*PWaveTimeStep()
qt.Controller()
V = qt.View()

# O.saveTmp()  # Save the simulation, that it can used later
# o.run()

# utils.waitIfBatch()  # wait, batch file execution
