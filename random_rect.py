# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.02.2021
# random_rect.py
# Desc: Create and plot a random rectangle, calculate characteristics

import random

import numpy as np
import matplotlib.pyplot as plt

from trockenmauer.utils import set_axes_equal



# x-dir: length 0.2
x = random.uniform(0.1, 0.3)
# y-dir: width 0.1
y = 1 * random.uniform(0.05, 0.15)
# z-dir: height 0.1
z = 0.5 * random.uniform(0.05, 0.15)


u = np.array([[x+1, y, z], [-x+1, y, z], [-x+1, -y, z], [x+1, -y, z],
              [x+1, y, -z], [-x+1, y, -z], [-x+1, -y, -z], [x+1, -y, -z]])
print(u.shape)

# pca of the points
cov_matrix = np.cov(u.T)
e_val, e_vec = np.linalg.eig(cov_matrix)
# print(e_val)
print(np.sqrt(e_val))
e_val_sqrt = np.sqrt(e_val)
print(np.argsort(e_val_sqrt))
# print(e_vec)

fig = plt.figure()
ax = plt.axes(projection='3d')
mean = np.mean(u, axis=0)
# why are the eigenvalues so big -> cov=square? -> with sqrt -> its almost the length of the cube
order = np.argsort(e_val)  # indizes to ordered eigenvalues (from small to big)
ordered_axes = [f'height {np.round(e_val_sqrt[order[0]], 2)}',
                f'width {np.round(e_val_sqrt[order[1]], 2)}',
                f'length {np.round(e_val_sqrt[order[2]], 2)}']
# texts = [f'length: {np.round(l, 1)}', f'width {np.round(w, 1)}', f'height {np.round(h, 1)}']
# texts = [f'{cube_axes[order[0]]} {np.round(e_val[0], 1)}', f'{cube_axes[order[1]]} {np.round(e_val[1], 1)}', f'{cube_axes[order[2]]} {np.round(e_val[2], 1)}', ]
texts = order

# Plot the points
ax.plot3D(np.append(u[:, 0], u[0, 0]), np.append(u[:, 1], u[0, 1]), np.append(u[:, 2], u[0, 2]), 'g.')
# Plot the eigenvectors
# plot the axes in order (shortest to longest)
ax.plot3D(mean[0], mean[1], mean[2], 'r.')
for o, direction in zip(order, ordered_axes):
    val = np.sqrt(e_val[o])
    vec = e_vec[o, :]
    print(vec)
    # m = mean[0]
    # ax.plot3D([m, val*vec[0]], [m, val*vec[1]], [m, val*vec[2]], 'r')
    ax.plot3D([mean[0], mean[0] + val*vec[0]], [mean[1], mean[1] + val*vec[1]], [mean[2], mean[2] + val*vec[2]], 'r')
    ax.text(mean[0] + val*vec[0], mean[1] + val*vec[1], mean[2] + val*vec[2], direction, 'x')


# Calculate the rotation matrix
def rot_matrix(l, w, h, x=np.array([1, 0, 0]), y=np.array([0, 1, 0]), z=np.array([0, 0, 1])) -> np.ndarray:
    # solve ax = b
    orig = np.vstack((l, w, h))
    goal = np.vstack((x, y, z))

    a = orig
    b = goal
    x = np.linalg.solve(a, b)
    return x
    # return np.reshape(x, (3, 3))


r = rot_matrix(e_vec[order[2]], e_vec[order[1]], e_vec[order[0]])

# u_rot = np.matmul(u, r)
ordered_eig = np.vstack((e_vec[order[2]], e_vec[order[1]], e_vec[order[0]]))
# print('eigen')
# print(np.round(e_vec, 1))
print('ordered eigen')
print(np.round(ordered_eig, 1))
print('rotation')
print(np.round(r, 1))
u_rot = np.matmul(u, r)
# print(u)
# print(u_rot)
# Plot the rotated points
ax.plot3D(np.append(u_rot[:, 0], u_rot[0, 0]), np.append(u_rot[:, 1], u_rot[0, 1]), np.append(u_rot[:, 2], u_rot[0, 2]), 'black')

# calculate the center
print(u_rot)
print(u_rot - np.mean(u_rot, axis=0))


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
set_axes_equal(ax)
plt.show()
