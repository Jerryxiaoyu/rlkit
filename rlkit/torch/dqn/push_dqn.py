from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import rlkit.torch.pytorch_util as ptu
from torch.autograd import Variable


class PushDQNTrainer(TorchTrainer):
    def __init__(
            self,
            qf,
            target_qf,
            learning_rate=1e-3,
            soft_target_tau=1e-3,
            target_update_period=1,
            qf_criterion=None,

            discount=0.99,
            reward_scale=1.0,
            ignore_keys =[],
    ):
        super().__init__()
        self.qf = qf
        self.target_qf = target_qf
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        #self.qf_optimizer = optim.Adam( self.qf.parameters(), lr=self.learning_rate,)

        self.qf_optimizer = torch.optim.SGD(self.qf.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.ignore_keys = ignore_keys

    def train_from_torch(self, batch):
        rewards = batch['rewards'] * self.reward_scale
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        next_obs = next_obs.reshape((next_obs.shape[0], 50,50,4))
        obs = obs.reshape((obs.shape[0], 50, 50, 4))

        n_batch = obs.shape[0]
        """
        Compute loss
        """
        change_detected = True
        # Compute future reward
        if not change_detected  :
            future_reward = 0
        else:
            next_color_heightmap = next_obs[:,:,:,:3]
            next_depth_heightmap = next_obs[:,:,:,-1]
            next_push_predictions, next_grasp_predictions, next_state_feat = self.target_qf(next_color_heightmap,
                                                                                           next_depth_heightmap,
                                                                                           is_volatile=True)
            future_reward = np.max(next_push_predictions, axis=(1,2,3)).reshape((n_batch,-1))
        #target_q_values = self.target_qf(next_obs).detach().max(1, keepdim=True )[0]

        y_target = rewards + (1. - terminals) * self.discount * ptu.from_numpy(future_reward)
        #y_target = y_target.detach()

        lable_size = 80  # 320
        action_size = 50  # 224
        padding_half = 15



        label = torch.zeros((n_batch, lable_size, lable_size)).float()
        specified_angle_index = []
        for i in range(n_batch):
            action_area = torch.zeros((action_size, action_size)).float()
            action_area[int(actions[i, 1].item())][int(actions[i, 2].item())] = 1

            tmp_label = torch.zeros((action_size, action_size)).float()
            tmp_label[action_area > 0] = y_target[i]
            label[i, padding_half:(lable_size - padding_half), padding_half:(lable_size - padding_half)] = tmp_label

            label_weights = torch.zeros(label.shape).float()
            tmp_label_weights = torch.zeros((action_size, action_size)).float()
            tmp_label_weights[action_area > 0] = 1
            label_weights[i, padding_half:(lable_size - padding_half),
            padding_half:(lable_size - padding_half)] = tmp_label_weights

            specified_angle_index.append(int(actions[i, 0].item()))


        specified_angle_index = np.array(specified_angle_index)
        # Compute loss and backward pass

        color_heightmap = obs[:, :, :, :3]
        depth_heightmap = obs[:, :, :, -1]
        _,_,_ = self.qf(color_heightmap,  depth_heightmap,  is_volatile=False,  specific_rotation= specified_angle_index)

        for i in range(n_batch):
            if i == 0:
                pf_pred = self.qf.model.output_prob[0][0].view(1, lable_size,  lable_size)

            else:
                pf_pred = torch.cat((pf_pred, self.qf.model.output_prob[i][0].view(1, lable_size,  lable_size)), dim = 0)

        loss = self.qf_criterion(pf_pred, Variable(label.cuda()* label_weights.cuda() , requires_grad=False))

        # actions is a one-hot vector
        # y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss =  loss.sum()
        """
        Soft target network updates
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )

        print('loss value:', np.mean(ptu.get_numpy(qf_loss)))
        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_target),
            ))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
        )
