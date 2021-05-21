# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.21, tester.py
#

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from trockenmauer.utils import set_axes_equal
from trockenmauer.math_utils import pca, rot_matrix, Rotation, Translation, RotationTranslation

# Test the rotation with random points

# For mathematical reasons: coordinates as column vectors
points = np.array([[10, 13, 6], [13, 9, 16], [15, 5, 8], [11, 7, 8], [6, 7, 13]]).T
# x, y, z = [3, 8, 5]
# a = [0, 0, 0]
# b = [x, 0, 0]
# c = [x, y, 0]
# d = [0, y, 0]
# e = [0, 0, z]
# f = [x, 0, z]
# g = [x, y, z]
# h = [0, y, z]
# points = np.array([a, b, c, d,  # lower 4 vertices
#               e, f, g, h]).T  # upper 4 vertices
center = points.mean(axis=1)

# eigenvectors and eigenvalues
cov_matrix = np.cov(points)
e_val, e_vec = np.linalg.eig(cov_matrix)
e_val = np.sqrt(e_val)

# change the order of the eigenvectors
order = np.argsort(e_val)[::-1]
e_vec = e_vec[:, order]
e_val = e_val[order]

scaled_e_vec = e_val * e_vec
# Transpose to origin
points_translated = points.T - center
points_translated = points_translated.T
center_translated = points_translated.mean(axis=1)

# Rotate with ev
r = e_vec.T
points_rotated = r @ points_translated
center_rotated = points_rotated.mean(axis=1)
# new ev
cov_matrix = np.cov(points_rotated)
e_val_2, e_vec_2 = np.linalg.eig(cov_matrix)
e_val_2 = np.sqrt(e_val_2)
scaled_e_vec_2 = e_val_2 * e_vec_2

# translate back
points_re_translated = points_rotated.T + center
points_re_translated = points_re_translated.T
center_re_translated = points_re_translated.mean(axis=1)
# new ev
cov_matrix = np.cov(points_rotated)
e_val_new, e_vec_new = np.linalg.eig(cov_matrix)
e_val_new = np.sqrt(e_val_new)
scaled_e_vec_new = e_val_new * e_vec_new

# plotting -> coordinates as row vectors -> Transposing everything
points = points.T
points_translated = points_translated.T
points_rotated = points_rotated.T
points_re_translated = points_re_translated.T
scaled_e_vec = scaled_e_vec  # !!! DONT TRANSPOSE THE EIGENVECTORS
scaled_e_vec_2 = scaled_e_vec_2
scaled_e_vec_new = scaled_e_vec_new

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# 4 original points
ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'g.', markersize=4)
# original center + ev
cx, cy, cz = points.mean(axis=0)
ax.plot3D(cx, cy, cz, 'r.', markersize=4)
for vec in scaled_e_vec.T:
    ax.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'r')

# translate center of points to origin
ax.plot3D(points_translated[:, 0], points_translated[:, 1], points_translated[:, 2], 'g.', markersize=4)
# center + ev
cx, cy, cz = points_translated.mean(axis=0)
ax.plot3D(cx, cy, cz, 'r.', markersize=4, alpha=0.2)
for vec in scaled_e_vec.T:
    ax.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'r', alpha=0.2)

# rotate the points around the center
ax.plot3D(points_rotated[:, 0], points_rotated[:, 1], points_rotated[:, 2], 'b.', markersize=4)
# center + ev
cx, cy, cz = points_rotated.mean(axis=0)
ax.plot3D(cx, cy, cz, 'r.', markersize=4)
for vec in scaled_e_vec_2.T:
    ax.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'r')

# translate back
ax.plot3D(points_re_translated[:, 0], points_re_translated[:, 1], points_re_translated[:, 2], 'b.', markersize=4)
# center + ev
cx, cy, cz = points_re_translated.mean(axis=0)
ax.plot3D(cx, cy, cz, 'b.', markersize=4)
for vec in scaled_e_vec_new.T:
    ax.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'b')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
set_axes_equal(ax)
# plt.show()

# ---------------------------------------------------------------------------------------------
# Test the rotation with random points
print('------------------------------------------------')

# For mathematical reasons: coordinates as column vectors
points = np.array([[10, 13, 6], [13, 9, 16], [15, 5, 8], [11, 7, 8], [6, 7, 13]]).T
# x, y, z = [3, 8, 5]
# a = [0, 0, 0]
# b = [x, 0, 0]
# c = [x, y, 0]
# d = [0, y, 0]
# e = [0, 0, z]
# f = [x, 0, z]
# g = [x, y, z]
# h = [0, y, z]
# points = np.array([a, b, c, d,  # lower 4 vertices
#               e, f, g, h]).T  # upper 4 vertices
center = points.mean(axis=1)

# eigenvectors and eigenvalues
e_val, e_vec = pca(points)
# change the order of the eigenvectors
order = np.argsort(e_val)[::-1]
r = rot_matrix(e_vec, order)
# change the eigenvectors / values by the new order
e_vec = e_vec[:, order]
e_val = e_val[order]
scaled_e_vec = e_val * e_vec

# Transpose to origin
transformation1 = Translation(translation=-center)
points_translated = transformation1.transform(points)
center_translated = points_translated.mean(axis=1)

# Rotate with ev
t = RotationTranslation(rotation=r, center=center, translation=-center)
points_rotated = t.transform(points)
center_rotated = points_rotated.mean(axis=1)

# new ev
cov_matrix = np.cov(points_rotated)
e_val_2, e_vec_2 = np.linalg.eig(cov_matrix)
e_val_2 = np.sqrt(e_val_2)
scaled_e_vec_2 = e_val_2 * e_vec_2

# translate back
t = Rotation(rotation=r, center=center)
points_re_translated = t.transform(points)

center_re_translated = points_re_translated.mean(axis=1)
# new ev
cov_matrix = np.cov(points_rotated)
e_val_new, e_vec_new = np.linalg.eig(cov_matrix)
e_val_new = np.sqrt(e_val_new)
scaled_e_vec_new = e_val_new * e_vec_new

# plotting -> coordinates as row vectors -> Transposing everything
points = points.T
points_translated = points_translated.T
points_rotated = points_rotated.T
points_re_translated = points_re_translated.T
scaled_e_vec = scaled_e_vec  # !!! DONT TRANSPOSE THE EIGENVECTORS
scaled_e_vec_2 = scaled_e_vec_2
scaled_e_vec_new = scaled_e_vec_new

fig2 = plt.figure()
ax2 = Axes3D(fig2, auto_add_to_figure=False)
fig2.add_axes(ax2)

# 4 original points
ax2.plot3D(points[:, 0], points[:, 1], points[:, 2], 'g.', markersize=4)
# original center + ev
cx, cy, cz = points.mean(axis=0)
ax2.plot3D(cx, cy, cz, 'r.', markersize=4)
for vec in scaled_e_vec.T:
    ax2.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'r')

# translate center of points to origin
ax2.plot3D(points_translated[:, 0], points_translated[:, 1], points_translated[:, 2], 'g.', markersize=4)
# center + ev
cx, cy, cz = points_translated.mean(axis=0)
ax2.plot3D(cx, cy, cz, 'r.', markersize=4, alpha=0.2)
for vec in scaled_e_vec.T:
    ax2.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'r', alpha=0.2)

# rotate the points around the center
ax2.plot3D(points_rotated[:, 0], points_rotated[:, 1], points_rotated[:, 2], 'b.', markersize=4)
# center + ev
cx, cy, cz = points_rotated.mean(axis=0)
ax2.plot3D(cx, cy, cz, 'r.', markersize=4)
for vec in scaled_e_vec_2.T:
    ax2.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'r')

# translate back
ax2.plot3D(points_re_translated[:, 0], points_re_translated[:, 1], points_re_translated[:, 2], 'b.', markersize=4)
# center + ev
cx, cy, cz = points_re_translated.mean(axis=0)
ax2.plot3D(cx, cy, cz, 'b.', markersize=4)
for vec in scaled_e_vec_new.T:
    ax2.plot3D([cx, cx + vec[0]], [cy, cy + vec[1]], [cz, cz + vec[2]], 'b')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
set_axes_equal(ax2)
plt.show()
