"""Heuristic push policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from rlkit.policies.base import Policy
from multiworld.envs.pybullet.sampler.heuristic_push_mask_sampler import HeuristicPushMaskSampler
from multiworld.envs.pybullet.util.bullet_camera import create_camera
import pybullet as p
import math
import numpy as np
from torch import nn
from multiworld.perception.depth_utils import scale_mask
import matplotlib.pyplot as plt


class HeuristicPushPolicy( Policy):
    """Heuristic push policy."""

    def __init__(self,

                 variant):
        """Initialize.

        Args:
            env: Environment.
            config: Policy configuration.
        """
        super().__init__()

        start_margin = 0.05,
        motion_margin = 0.01,
        max_attemps = 20000

        num_object = 1
        camera_params = variant.get('camera_params', None)
        self.num_objects = variant.get('num_objects', 1)
        cspace_high = variant.get('cspace_high', (0 + 0.25, -0.40 + 0.15, 0.154))
        cspace_low = variant.get('cspace_low', (0 - 0.25, -0.40 - 0.15, 0.154))
        mask_bias = variant.get('mask_bias', 2)
        mask_scale = variant.get('mask_scale', 1.1)
        heightmap_resolution = variant.get('heightmap_resolution', 0.008)

        self._camera = create_camera(p,
                               camera_params['image_height'],
                               camera_params['image_width'],
                               camera_params['intrinsics'],
                               camera_params['translation'],
                               camera_params['rotation'],
                               near=camera_params['near'],
                               far=camera_params['far'],
                               distance=camera_params['distance'],
                               camera_pose=camera_params['camera_pose'],

                               is_simulation=True)

        self.cspace_low = np.array(cspace_low)
        self.cspace_high = np.array(cspace_high)
        self.cspace_offset = 0.5 * (self.cspace_high + self.cspace_low)
        self.cspace_range = 0.5 * (self.cspace_high - self.cspace_low)

        self.workspace_limits = np.asarray([
            [self.cspace_low[0], self.cspace_high[0]],  # x
            [self.cspace_low[1], self.cspace_high[1]],  # y
            [0.035, 0.4]])

        self.heightmap_resolution = heightmap_resolution
        self.heatmap_shape = np.round(
            ((self.workspace_limits[1][1] - self.workspace_limits[1][0]) / self.heightmap_resolution,
             (self.workspace_limits[0][1] - self.workspace_limits[0][0]) / self.heightmap_resolution)).astype(
            int)


        ## hard code!!!   for ottermodel , mask bias =2 ; for multiworld , mask bias =3
        self.mask_body_index_list = [_ + mask_bias for _ in range(num_object)]


        self.start_margin = start_margin
        self.motion_margin = motion_margin
        self.max_attemps = max_attemps

        self.last_end = None

        self.PUSH_DELTA_SCALE_X = self.PUSH_DELTA_SCALE_Y = 0.1
        self.PUSH_MIN = 0.01
        self.PUSH_MAX = 0.1
        self.PUSH_SCALE = 2.
        self.MASK_MARGIN_SCALE = mask_scale

        self.num_object = num_object

        self.vis_debug = False
        if self.vis_debug:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            plt.ion()
            plt.show()
            self.ax = ax


    def _convert_mask(self, segmask, body_mask_index):
        mask = segmask.copy()
        mask[mask != self.mask_body_index_list[body_mask_index]] = 0
        mask[mask == self.mask_body_index_list[body_mask_index]] = 255

        scaled_mask = scale_mask(mask, self.MASK_MARGIN_SCALE, value=self.mask_body_index_list[body_mask_index])

        return scaled_mask

    def find_in_array(self, to_find_v, to_find_u, in_arr):
        """
        find if a point [u,v] exsits in a 2-dim in_arr

        :param to_find_u:
        :param in_arr:
        :return:
        """

        v_found = np.where(in_arr[:, 0] == to_find_v)
        if len(v_found[0]) > 0:
            u_found = np.where(in_arr[v_found][:, 1] == to_find_u)
            if len(u_found[0]) > 0:
                return True

        return False

    def _sample(self, segmask, depth, body_mask_index):
        # get a point from object mask
        scale_mask = self._convert_mask(segmask, body_mask_index)

        idx = np.argwhere(scale_mask == self.mask_body_index_list[body_mask_index])

        action_pixel = 0
        for _ in range(self.max_attemps):
            sampled_index = np.random.randint(0, idx.shape[0], size=1)

            sampled_points = idx[sampled_index]
            sampled_points = sampled_points[:, ::-1]  ## v, u --> u, v

            base_point = sampled_points[0]  ##  u, v

            if depth.dtype == np.uint16:
                depth = depth / 1000.
            #bp_w = self._camera.deproject_pixel(base_point, depth[base_point[1], base_point[0]], is_world_frame=True)[:2]


            dis = 45  # pixels  coresp. 0.05m

            num_rotations = 16
            points = []
            for i in range(num_rotations):
                theta = 2 * np.pi / num_rotations * i
                u = max(min(int(base_point[0] + np.cos(theta) * dis), 640-1) ,0)
                v = max(min(int(base_point[1] + np.sin(theta) * dis),480-1),0)
                points.append(np.array([u, v]))

            points = np.array(points)  # u, v

            in_segmask_points = np.array([self.find_in_array(points[i, 1], points[i, 0], idx) for i in range(num_rotations)])

            not_in_segmaks_index = np.where(in_segmask_points == False)[0]

            if len(not_in_segmaks_index) > 0:
                direction = np.random.choice(not_in_segmaks_index)
                action_pixel = points[direction]  # u, v
                break

        action_pos = self._camera.deproject_pixel(action_pixel, depth[action_pixel[1], action_pixel[0]], is_world_frame=True)[ :2]

        pix_u = (action_pos[0] - self.workspace_limits[0][0]) / self.heightmap_resolution
        pix_v = (action_pos[1] - self.workspace_limits[1][0]) / self.heightmap_resolution

        pix_u = max(min(int(np.ceil(pix_u)), self.heatmap_shape[1]-1), 0)
        pix_v = max(min(int(np.ceil(pix_v)), self.heatmap_shape[0]-1), 0)
        angle = (16- (direction+8)%16)%16

        if self.vis_debug:
            self.plot_result(segmask, base_point, points, direction)

        return np.array([angle, pix_v, pix_u])


    def get_action(self, obs):
        """Implementation of action.
        Args:
            observation: The observation of the current step.

        Returns:
            action: The action of the current step.
        """

        segmask = obs['mask_observation']
        depth = obs['depth_observation']

        body_index =  0#np.random.randint(0, self.num_objects)
        action = self._sample(segmask, depth,
                                      body_mask_index=body_index,
                                      )

        return action, {}

    def plot_result(self, image, base_point, points, angle):


        self.ax.cla()
        self.ax.imshow(image)

        self.ax.scatter(base_point[0], base_point[1], color='g', s=50)
        for i in range(16):
            if i == angle:
                c = 'y'
            else:
                c = 'r'
            self.ax.scatter(points[i, 0], points[i, 1], color=c, s=50)

        ##-----
        plt.draw()
        plt.pause(1e-3)
