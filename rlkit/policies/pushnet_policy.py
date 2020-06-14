"""
Torch argmax policy
"""
import numpy as np
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


from rlkit.util.push_utils import get_prediction_vis
import cv2

class PushNetArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
       # obs = np.expand_dims(obs, axis=0)
       # obs = ptu.from_numpy(obs).float()

        # Run forward pass with network to get affordances
        push_predictions, grasp_predictions, state_feat = self.qf(obs['color_heatmap'], obs['valid_depth_heightmap'],
                                                                          is_volatile=True)


        # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)

        best_action = np.unravel_index(np.argmax(push_predictions[0]), push_predictions[0].shape)
        #predicted_value = np.max(push_predictions)


        # img = get_prediction_vis(push_predictions[0], obs['color_heatmap'], best_action )
        #
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return np.array(best_action), {}