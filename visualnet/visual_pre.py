from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import namedtuple
from got10k.trackers import Tracker
from .backbones import AlexNetV1
from .heads import SiamFC

__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, device_id, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device_id}' if self.cuda else 'cpu')
        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 1080,
            'context': 0.5,
            # inference parameters
            'scale_num': 4,
            'scale_step': 1.0375,
            'scale_lr': 0.69,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 80,
            'response_up': 4,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,
            'new_len': 120,  # size of square img
            'x_sz': 440
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, examplar, box):
        '''
        crop the template and extract the feature as kernel. return the kernel
        '''
        self.net.eval()  # set to evaluation mode

        # refGT is 0-indexed [x, y, w, h ---> center based [y, x, h, w]
        box = np.array([
            box[1] + box[3] / 2,
            box[0] + box[2] / 2,
            box[3], box[2]], dtype=np.float32)
        self.target_sz = box[2:]
        self.z_sz = np.max(self.target_sz)
        self.scale_factors = np.array([ 0.75, 1, 1.5])
        # exemplar features
        z = torch.from_numpy(examplar).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
        self.scale = 1

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()
        # x: is a square img
        self.x_sz = img.shape[0]
        self.instance_sz = self.x_sz * self.cfg.exemplar_sz / self.z_sz

        # responses
        responses_list = []
        for i in range(len(self.scale_factors)):
            x_ins = cv2.resize(img, (int(self.instance_sz * self.scale_factors[i]),
                                     int(self.instance_sz * self.scale_factors[i])),
                               interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(x_ins).to(
                self.device).permute(2, 0, 1).unsqueeze(0).float()
            x = self.net.backbone(x)

            responses = self.net.head(self.kernel, x)
            responses = responses.squeeze(0).squeeze(0).cpu().numpy()
            responses_list.append(responses)

        # peak scale
        scale_id = np.argmax([np.max(responses_list[i]) for i in range(len(self.scale_factors))])
        response_org = responses_list[scale_id]
        # upsample responses and penalize scale changes
        self.upscale_sz = int(self.cfg.response_up * response_org.shape[1])
        response = cv2.resize(
            response_org, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
        # peak location
        response -= response.min()
        response /= response.sum() + 1e-16
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz \
                        / int(self.instance_sz * self.scale_factors[scale_id])
        img_center = np.array([img.shape[0] / 2 - 1, img.shape[1] / 2 - 1])
        loc_img = img_center + disp_in_image
        re_sz = int(response_org.shape[0] * self.cfg.total_stride * self.x_sz \
                    / int(self.instance_sz * self.scale_factors[scale_id]))
        response = cv2.resize(response_org, (re_sz, re_sz))

        # update target size
        target_sz = self.target_sz / self.scale_factors[scale_id]
        # 0-indexed and left-top based bounding box (sample coordinate)
        box = np.array([
            loc_img[1] - (target_sz[1] - 1) / 2,
            loc_img[0] - (target_sz[0] - 1) / 2,
            target_sz[1], target_sz[0]])  # box for x_crop
        img_center = np.array([(img.shape[0] - 1) / 2, (img.shape[0] - 1) / 2])
        corners = np.concatenate(
            (np.round(img_center - (response.shape[0]) / 2),
             np.round(img_center + (response.shape[0]) / 2)))
        corners = np.round(corners).astype(int)
        img_crop = img[corners[0]:corners[2], corners[1]:corners[3]]
        re_box = np.array([
            corners[0], corners[1],
            response.shape[0], response.shape[0]
        ])  # [x,y,w,h]box of response map (in sample coordinate)
        return box, re_box, self.scale_factors[scale_id]

    def observ(self, examplar, refGT, sample_img):

        self.init(examplar, refGT)
        box, re_box, scale_id = self.update(sample_img)

        return box, re_box, scale_id

