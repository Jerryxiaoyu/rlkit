#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time


import torch.utils.data

class pushModel(nn.Module):

    def __init__(self, use_cuda, n_batch = 32):  # , snapshot=None
        super(pushModel, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

        self.n_batch = n_batch

        self.batch_flow_grid_before_list, self.batch_flow_grid_after_list = self.get_flow_grid(self.n_batch)
        self.single_flow_grid_before_list, self.single_flow_grid_after_list = self.get_flow_grid(1)


    def get_flow_grid(self, n_batch):

        input_data_size = torch.Size((n_batch, 3, 160, 160))
        interm_push_feat_size = torch.Size((n_batch, 2048, 5, 5))
        flow_grid_before_list = []
        flow_grid_after_list =[]
        for rotate_idx in range(self.num_rotations):
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                            [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)

            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

            affine_mat_before = affine_mat_before.repeat((n_batch, 1, 1))
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_data_size)
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_data_size)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_mat_after = affine_mat_after.repeat((n_batch, 1, 1))
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_push_feat_size)
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_push_feat_size)


            flow_grid_before_list.append(flow_grid_before)
            flow_grid_after_list.append(flow_grid_after)

        return flow_grid_before_list,flow_grid_after_list


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        n_batch = input_color_data.shape[0]
        if is_volatile:
            output_prob = []
            interm_feat = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):

                if n_batch == 1:
                    flow_grid_before = self.single_flow_grid_before_list[rotate_idx]
                    flow_grid_after = self.single_flow_grid_after_list[rotate_idx]
                elif n_batch == self.n_batch:
                    flow_grid_before = self.batch_flow_grid_before_list[rotate_idx]
                    flow_grid_after = self.batch_flow_grid_after_list[rotate_idx]
                else:
                    raise NotImplementedError

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before,
                                                 mode='nearest')

                else:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                 mode='nearest')

                # Compute intermediate features
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

                interm_feat.append([interm_push_feat, ])

               # a = F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                    ])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            for batch_index, rotate_idx  in  enumerate(specific_rotation):
                #rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))


                flow_grid_before = self.single_flow_grid_before_list[rotate_idx]
                flow_grid_after = self.single_flow_grid_after_list[rotate_idx]



                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(Variable(input_color_data[batch_index].unsqueeze(0), requires_grad=False).cuda(), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data[batch_index].unsqueeze(0), requires_grad=False).cuda(), flow_grid_before,
                                                 mode='nearest')
                else:
                    rotate_color = F.grid_sample(Variable(input_color_data[batch_index].unsqueeze(0), requires_grad=False), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data[batch_index].unsqueeze(0), requires_grad=False), flow_grid_before,
                                                 mode='nearest')

                # Compute intermediate features
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

                self.interm_feat.append([interm_push_feat, ])



                # Forward pass through branches, undo rotation on output predictions, upsample results
                self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                         ])

            return self.output_prob, self.interm_feat


class PushNet(nn.Module):
    def __init__(self, use_cuda, n_batch):  # , snapshot=None
        super(PushNet, self).__init__()
        self.model = pushModel(use_cuda, n_batch = n_batch)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()


    def forward(self, color_heightmap, depth_heightmap, is_volatile=True,  specific_rotation=-1):

        if len(color_heightmap.shape) ==3 and len(depth_heightmap.shape) ==2:
            color_heightmap = color_heightmap[None]
            depth_heightmap = depth_heightmap[None]
        elif len(color_heightmap.shape) ==4 and len(depth_heightmap.shape) ==3:
            pass
        else:
            raise NotImplementedError


        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[1, 2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[1, 2, 2], order=0)
        assert (color_heightmap_2x.shape[1:3] == depth_heightmap_2x.shape[1:3])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[1]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[1]) / 2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, :, 0],
                                      ((0, 0), (padding_width, padding_width), (padding_width, padding_width)),
                                      'constant', constant_values=0)
        color_heightmap_2x_r.shape = (
        color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], color_heightmap_2x_r.shape[2], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, :, 1],
                                      ((0, 0), (padding_width, padding_width), (padding_width, padding_width)),
                                      'constant', constant_values=0)
        color_heightmap_2x_g.shape = (
        color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], color_heightmap_2x_r.shape[2], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, :, 2],
                                      ((0, 0), (padding_width, padding_width), (padding_width, padding_width)),
                                      'constant', constant_values=0)
        color_heightmap_2x_b.shape = (
        color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], color_heightmap_2x_r.shape[2], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=3)
        depth_heightmap_2x = np.pad(depth_heightmap_2x,
                                    ((0, 0), (padding_width, padding_width), (padding_width, padding_width)),
                                    'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, :, c] = (input_color_image[:, :, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (
        depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], depth_heightmap_2x.shape[2], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=3)
        for c in range(3):
            input_depth_image[:, :, :, c] = (input_depth_image[:, :, :, c] - image_mean[c]) / image_std[c]

        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(0, 3, 1, 2)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(0, 3, 1, 2)

        # Pass input data through model
        output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:, :,
                                   int(padding_width / 2):int(color_heightmap_2x.shape[1] / 2 - padding_width / 2),
                                   int(padding_width / 2):int(color_heightmap_2x.shape[1] / 2 - padding_width / 2)]
                # grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
            else:
                push_predictions = np.concatenate((push_predictions,
                                                   output_prob[rotate_idx][0].cpu().data.numpy()[:, :,
                                                   int(padding_width / 2):int(
                                                       color_heightmap_2x.shape[1] / 2 - padding_width / 2),
                                                   int(padding_width / 2):int(
                                                       color_heightmap_2x.shape[1] / 2 - padding_width / 2)]),
                                                  axis=1)
                    # grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
        # grasp_predictions
        return push_predictions, 0, state_feat
