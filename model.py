import numpy as np
import torch
import torch.nn as nn
import torchvision

from pre_proc import Config

opt = Config()
N_CLASS = 1


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


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        rawnet = torchvision.models.vgg16(pretrained=True)
        self.train_trans = False
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        self.roipool = SlowROIPool(output_size=(7, 7))
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])
        self.cls_score = nn.Linear(4096, N_CLASS + 1)
        self.bbox = nn.Linear(4096, 4)
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()
        if opt.attention:
            self.attn = nn.Sequential(
                nn.Conv2d(512, 128, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid(),
            )
            self.mse = nn.MSELoss(reduction='sum')
        '''self.trans = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
        )
        self.cls_trans = nn.Sequential(*list(rawnet.classifier.children())[:-1],
                                       nn.Linear(4096, N_CLASS + 1))'''

    def forward(self, inp, rois, ridx, type):  # , labels):
        res = self.seq(inp)
        attn = torch.ones(res.size()).cuda()
        if opt.attention:
            attn = self.attn(res).repeat(1, res.size(1), 1, 1)
        res = res * attn
        res = self.roipool(res, rois, ridx)

        if type == 'train_rcnn':
            res = res.view(res.size(0), -1)
            feat = self.feature(res)
            cls_score = self.cls_score(feat)
            bbox = self.bbox(feat).view(-1, 1, 4).repeat(1, 2, 1)
            # bbox = self.bbox(feat).view(-1, 2, 4)
            return cls_score, bbox, attn[:, 0]
        elif type == 'train_trans':
            res = res.detach()
            trans = res + self.trans(res)
            flat = trans.view(trans.size(0), -1)
            cls_score = self.cls_trans(flat)
            return cls_score, trans
        elif type == 'test':
            if opt.transformation:
                trans = res + self.trans(res)
                flat = trans.view(trans.size(0), -1)
                cls_score = self.cls_trans(flat)
                res = res.view(res.size(0), -1)
                bbox = self.bbox(self.feature(res)).view(-1, 1, 4).repeat(1, 2, 1)
            else:
                res = self.feature(res.view(res.size(0), -1) )
                cls_score = self.cls_score(res)
                bbox = self.bbox(res).view(-1, 1, 4).repeat(1, 2, 1)
            return cls_score, bbox
        else:
            raise ('In rcnn.forward(): Only three options allowed: train_rcnn, train_trans, test.')

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

    def calc_loss_trans(self, probs, trans, labels):
        loss_sc = self.cel(probs, labels)
        lbl = labels.view(-1, 1, 1, 1).repeat(1, 512, 7, 7)
        trans_target = lbl * self.center_pos + (1 - lbl) * self.center_neg
        trans = trans.view(-1, 512 * 7 * 7)
        trans_target = trans_target.view(-1, 512 * 7 * 7)
        loss_trans = self.sl1(trans, trans_target)
        loss = loss_sc + loss_trans * opt.lmb_trans
        return loss, loss_sc, loss_trans

    def calc_center(self, inp, pos_rois, neg_rois, ridx):
        res = self.seq(inp)
        attn = torch.ones(res.size()).cuda()
        if opt.attention:
            attn = self.attn(res).repeat(1, res.size(1), 1, 1)
        res = res * attn
        pos = self.roipool(res, pos_rois, ridx)
        neg = self.roipool(res, neg_rois, ridx)
        pos = torch.sum(pos, 0)
        neg = torch.sum(neg, 0)
        return pos, neg

    def set_trans(self, center_pos, center_neg):
        self.train_trans = True
        self.center_pos = center_pos
        self.center_neg = center_neg
        #initialize three linear layers
        '''self.cls_trans[0].weight.data = self.feature[0].weight.data
        self.cls_trans[0].bias.data = self.feature[0].bias.data
        self.cls_trans[3].weight.data = self.feature[3].weight.data
        self.cls_trans[3].bias.data = self.feature[3].bias.data
        self.cls_trans[6].weight.data = self.cls_score.weight.data
        self.cls_trans[6].bias.data = self.cls_score.bias.data'''
