# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 02.03.21, test_sim.py
#

import random

from yade import utils
from yade.wrapper import Omega, CpmMat, Polyhedra
from yade import Polyhedra
from yade.polyhedra_utils import polyhedra
import numpy as np

from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.math_utils import transform

# place stones within x [0, 2], y [0, 1] -> 2m long, 1m wide
wall_x0, wall_x1 = [0, 2]
wall_y0, wall_y1 = [0, 1]

stone = generate_regular_stone(0.2, 0.1, 0.1)
# place the stone random within th wall
x = random.uniform(wall_x0, wall_x1)
y = random.uniform(wall_y0, wall_y1)
z = 1
# t = np.array([x, y, z])
# stone.transform(t=t)

# Omega is the scene?
o = Omega()

# surface at the bottom
o.bodies.append(utils.wall(0, axis=2, sense=1, material=m))

# add a concrete material
# mat = CpmMat()
mat = PolyhedraMat()
mat.density = 2600  # kg/m^3
mat.young = 1E6  # Pa
mat.poisson = 20000 / 1E6
mat.frictionAngle = 0.6  # rad
print(mat.dict())  # print all attributes

o.materials.append(mat)
p = polyhedra(mat, v=stone.vertices)
p.state.pos = (x, y, z)
print(p.dict())
print(Polyhedra.GetCentroid(p))
