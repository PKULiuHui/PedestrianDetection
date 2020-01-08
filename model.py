import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from pre_proc import Config

opt = Config()


class SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size

    def forward(self, images, rois, roi_idx):
        n = rois.shape[0]
        h = images.size(2)  # 30, original 480
        w = images.size(3)  # 40, original 640

        # Proposal bbox use [640, 480], but Network input use [480, 640]
        x1 = rois[:, 0]
        y1 = rois[:, 1]
        x2 = rois[:, 2]
        y2 = rois[:, 3]

        # x * w, y * h is correct, don't need to change!
        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)

        # print(images.shape, roi_idx)
        res = []
        for i in range(n):
            img = images[roi_idx[i]].unsqueeze(0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            img = self.maxpool(img)
            res.append(img)
        res = torch.cat(res, dim=0)
        return res


N_CLASS = 1


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        rawnet = torchvision.models.vgg16(pretrained=True)
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        self.roipool = SlowROIPool(output_size=(7, 7))
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])
        """
        self.feature = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )"""
        self.cls_score = nn.Linear(4096, N_CLASS + 1)
        self.bbox = nn.Linear(4096, 4)
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()
        if opt.attention:
            self.attn = nn.Sequential(
                nn.Conv2d(512, 128, 5, padding=2),
                nn.Conv2d(128, 128, 5, padding=2),
                nn.Conv2d(128, 128, 5, padding=2),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid(),
            )
            self.mse = nn.MSELoss(reduction='sum')

    def forward(self, inp, rois, ridx):  # , labels):
        res = self.seq(inp)
        attn = torch.ones(res.size()).cuda()
        if opt.attention:
            attn = self.attn(res).repeat(1, res.size(1), 1, 1)
        res = res * attn
        res = self.roipool(res, rois, ridx)
        res = res.view(res.size(0), -1)
        feat = self.feature(res)

        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, 1, 4).repeat(1, 2, 1)
        # bbox = self.bbox(feat).view(-1, 2, 4)
        return cls_score, bbox, attn[:, 0]

    def calc_loss(self, probs, bbox, attns, labels, gt_bbox, gt_attns):
        loss_sc = self.cel(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        loss_a = loss_sc * 0
        if opt.attention:
            loss_a = self.mse(attns, gt_attns) / attns.size(0)
        loss = loss_sc + opt.lmb_loc * loss_loc + opt.lmb_attn * loss_a
        return loss, loss_sc, loss_loc, loss_a
