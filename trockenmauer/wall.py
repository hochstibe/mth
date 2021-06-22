# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 27.05.21, wall.py
#
from typing import List, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
import time

import rtree.index
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .plot import set_axes_equal
from .generate_stones import generate_regular_stone
from . import NormalStone

if TYPE_CHECKING:
    from . import Stone, Boundary


class Wall:
    """
    A wall consists of the placed stones and the boundary.
    """
    boundary: 'Boundary'
    # mesh: 'pymesh.Mesh' = None -> not needed (merging in pymesh connects the meshes
    stones: List['Stone'] = []    # List of placed stones Todo: Why [0, ]???
    stones_vis: List['Stone'] = []  # List of valid and invalid placed stones

    # List of available stones
    normal_stones: List['Stone'] = []
    filling_stones: List['Stone'] = []

    r_tree: 'rtree.index.Index'  # The index of the aabb are the indices of wall.stones

    # The wall is built in levels: the algorithm tries to build one level after another
    level: int = 0  # enumeration (index) of the current building level
    level_h: List[List] = [[0, .1], ]  # List of [min z, max z] of the building levels
    level_free: float = 1  # ratio of free area on the current level
    _level_area: float = None  # total area on the current level (dependant o batter)

    def __init__(self, boundary: 'Boundary', stones: List['Stone'] = None):  # , mesh: 'pymesh.Mesh' = None):
        self.boundary = boundary
        self.stones = stones
        # self.mesh = mesh  -> not needed

        # Bounding Volume Hierarchy (axis aligned bounding boxes)
        # Use separate trees for different stone sizes (e.g. one for normal stones, one for filler stones)
        p = rtree.index.Property(dimension=3)
        self.r_tree = rtree.index.Index(properties=p)

        if not self.stones:
            self.stones = []

        # private attributes for the replay
        self._fig = None
        self._ax = None
        self._stone_frames = None  # lookup to find the stone based on the frame index

    def init_stones(self, n_normal_stones: int,  normal_x_range: Tuple[float, float],
                    normal_y_range: Tuple[float, float], normal_z_range: Tuple[float, float],
                    random: 'np.random.Generator', stone_type: str):

        # Generate and align stones to the coordinate axis
        stones = [generate_regular_stone(normal_x_range, normal_y_range, normal_z_range, random, stone_type)
                  for _ in range(n_normal_stones)]
        if stone_type.lower() == 'normal':
            self.normal_stones = stones
            # Order by their volume
            self.normal_stones.sort(key=lambda x: x.aabb_volume, reverse=True)
        elif stone_type.lower() == 'filling':
            self.filling_stones = stones
            # Order by their volume
            self.filling_stones.sort(key=lambda x: x.aabb_volume, reverse=True)
        else:
            raise ValueError

        # the first level is up to the max height of the available stones
        self.level_h[0] = [0, np.max([stone.height for stone in self.normal_stones])]

    @property
    def level_area(self):
        """
        total area on the current level (dependant of batter)
        """
        if not self._level_area:
            self._level_area = self.boundary.x * (self.boundary.y - 2*self.level_h[self.level][0]*self.boundary.batter)
        return self._level_area

    def add_stone(self, stone: 'Stone', invalid_color='red'):
        """
        Add a stone the the wall. If the stone intersects, it is only added vor visualization purposes

        :param stone: new stone object
        :param invalid_color: name of the color for an intersecting stone
        """
        # Add the stone to the mesh
        # if not self.mesh:
        #     self.mesh = stone.mesh
        # else:
        #     # Merging the mesh
        #     # self.mesh = pymesh.merge_meshes([self.mesh, stone.mesh])
        #     # self.mesh = pymesh.boolean(self.mesh, stone.mesh, 'union', engine='cgal')  # alternative to merge
        #     # -> for the moment, to check intersections, all individual stones have to be checked

        # Add the stone to the list to keep track of the individual stones
        self.stones_vis.append(stone)

        if stone.best_firefly and stone.best_firefly.validation_result.intersection:
            # stone intersects, don't add to tree
            stone.color = invalid_color
            stone.alpha = .1
        else:
            # valid stone
            stone.alpha = .9
            self.stones.append(stone)
            i = len(self.stones) - 1  # index of the stone
            # Add the BB to the tree, the name of the stone is the index in the stones-list
            self.r_tree.insert(i, stone.aabb_limits.flatten())

            level = self.get_stone_level(stone)
            if level == self.level and isinstance(stone, NormalStone):
                # update the free area for the normal stones. Filling stones can be above other stones
                # therefore, the area would be 2x subtracted
                self.level_free -= stone.aabb_area / self.level_area

    def get_stone_level(self, stone: 'Stone') -> Optional[int]:
        """
        Get the index of the building level for a (new) stone

        :param stone: new stone object
        """
        # returns 999, if no matching level found (above)
        z = stone.aabb_limits[1][2]
        limits = [lim for lim in self.level_h if lim[0] < z <= lim[1]]
        if limits:
            return self.level_h.index(limits[0])
        else:
            return 999

    def update_level_limits(self) -> bool:
        """
        Update h_max of the current level to the top of of the highest stone
        """
        h_min_old, h_max_old = self.level_h[self.level]
        h_max_new = np.max([stone.aabb_limits[1][2] for stone in self.stones])
        print(f'level {self.level}: {self.level_h[self.level]} min {h_min_old}, max {h_max_old}')
        if len(self.normal_stones) == 0:
            print('  no more stones available, cant pop one')
            return False
        if np.all(h_max_new == h_min_old):
            print('  !!! no stone placed on the current level - remove the biggest stone')
            print('  no level limits changed')
            self.normal_stones.pop(0)
            return False
        else:
            self.level_h[self.level] = [h_min_old, h_max_new]
            print('Updating the current level limits to', self.level_h[self.level])
            return True

    def next_level(self) -> bool:
        """
        Calculate the boundaries of the current building level (z_max = highest stone).
        Start a new building level. It returns a status code

        :return: True: New level started, or retry current level
                 False: Top of the wall reached or no more normal stones available, no new level started
        """
        # set h_max to the highest placed stone
        # returns True, while h_max is lower than the wall's limits
        h_min_old = self.level_h[self.level][0]
        h_max_old = np.max([stone.aabb_limits[1][2] for stone in self.stones])
        print(f'level {self.level}: {self.level_h[self.level]} min {h_min_old}, max {h_max_old}')
        # if no stone was placed on the previous level, no updating is needed
        if len(self.normal_stones) == 0:
            print('No more normal stones available')
            return False
        if np.all(h_max_old == h_min_old):
            print('no stone placed, dont go to next level (stone was removed with update_level_limits')
            return True
        else:
            # update old level
            self.level_h[self.level][1] = h_max_old
            # new level
            self.level += 1
            h_min_new = h_max_old
            h_max_new = h_max_old + np.max([stone.aabb_limits[1][2] for stone in self.normal_stones])
            if h_max_new > self.boundary.z:
                print('Top of the wall reached')
                return False
            self.level_h.append([h_min_new, h_max_new])
            print(f'new level {self.level}: {self.level_h[self.level]}')

        return True

    def _init(self):
        """
        Initializing the replay plot

        :return:
        """
        self._ax.clear()
        self.boundary.add_shape_to_ax(self._ax)
        # Set plot properties
        set_axes_equal(self._ax)
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('z')
        plt.draw()

    def _animate(self, i):
        self.stones_vis[i].add_shape_to_ax(self._ax)

    def _animate_fireflies(self, frame):
        """
        Animating the replay plot. For each stone, show

        * firefly iterations
        * final position

        """
        # hold still at the first frame
        if frame == 0:
            time.sleep(2)
        # Find the stone for the current frame
        stone_index = [i for i in self._stone_frames if frame in self._stone_frames[i]][0]
        stone = self.stones_vis[stone_index]
        hist = self.stones_vis[stone_index].position_history

        # plot previous stones
        self._ax.clear()
        self.boundary.add_shape_to_ax(self._ax)
        set_axes_equal(self._ax)
        for i in range(stone_index):
            self.stones_vis[i].add_shape_to_ax(self._ax)

        # check, if the frame should animate a firefly iteration or a stone
        history_index = frame - self._stone_frames[stone_index].start
        if history_index in range(len(hist)):
            # plot the history
            pos = hist[history_index].positions
            vel = hist[history_index].velocities

            self._ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', marker='.', s=20)
            self._ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2],
                            arrow_length_ratio=.1, linewidths=.5,
                            color='red')

        else:
            # after the fireflies, the final position of the stone is plotted
            stone.add_shape_to_ax(self._ax)

    def replay(self, fireflies: bool = False, save: Optional[str] = None, folder: str = '/home/stefan/mth/plots'):
        """
        Replays the building of the wall

        :return:
        """
        anim_running = True

        def on_space_press(event):
            nonlocal anim_running
            if event.key == ' ':
                if anim_running:
                    # anim.event_source.stop()
                    anim.pause()
                    anim_running = False
                else:
                    # anim.event_source.start()
                    anim.resume()
                    anim_running = True

        self._fig = plt.figure(figsize=(12, 9))
        self._ax = Axes3D(self._fig, auto_add_to_figure=False)
        self._fig.add_axes(self._ax)
        # Pause the animation on space press
        self._fig.canvas.mpl_connect('key_press_event', on_space_press)

        if fireflies:
            # calculate frames: number of stones and for each stone, the number of firefly positions
            frames = len(self.stones_vis) + sum([len(stone.position_history) for stone in self.stones_vis])
            # lookup dictionary to find the stone for any frame
            self._stone_frames = {}
            frame = 0
            for i in range(len(self.stones_vis)):
                num_iterations = len(self.stones_vis[i].position_history)
                self._stone_frames[i] = range(frame, frame + 1 + num_iterations)
                frame += num_iterations + 1

            # calculate the velocities for the fireflies
            for stone in self.stones_vis:
                hist = stone.position_history
                # vel = [hist[i + 1].positions - hist[i].positions for i in range(len(hist) - 1)]
                vel = list()
                for i in range(len(hist) - 1):
                    t1 = hist[i + 1].positions.copy()
                    t0 = hist[i].positions
                    if len(t1) < len(t0):
                        t1 = np.vstack((t1, t0[len(t1):]))
                        # print('add more dimensions to ti+1. Dim at t-1', vel[-1].shape, 'dim at t', (t1-t0).shape)
                    vel.append(t1 - t0)
                    # print(len(hist[i].positions), len(vel[i]))

                if vel:
                    # vel.insert(0, np.zeros((len(hist[0].positions), 3)))  # zero velocity at first iteration
                    vel.append(np.zeros((len(hist[-1].positions), 3)))
                for iteration, velocity in zip(stone.position_history, vel):
                    iteration.velocities = np.array(velocity)

            # Execute the animation
            anim = FuncAnimation(self._fig, self._animate_fireflies, frames, self._init, interval=500, repeat=True)

        else:
            # only plot the stones
            frames = len(self.stones_vis)
            anim = FuncAnimation(self._fig, self._animate, frames, self._init, interval=1000, repeat=True)

        plt.show()
        if save:
            folder = Path(folder)
            file = folder / f'{save}'
            print('Saving file to', file, '...')
            anim.save(f'{file}.gif', writer='pillow')
            print('  GIF saved')
            anim.save(f'{file}.mp4', writer=FFMpegWriter(fps=2))
            print('  MP4 saved')

    def __repr__(self):
        return f'<Wall(boundary={self.boundary}, stones={self.stones})>'