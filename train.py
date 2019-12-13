import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from model import *
from utils import *
from tqdm import trange
from pre_proc import *
import sys

#sys.path.insert(0, './evaluate')
#import evaluate

N_CLASS = 1

Transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

rcnn = RCNN().cuda()
print(rcnn)

opt = Config()
train_anno = { x : opt.annotation[x] for x in opt.annotation if int(x[3:5]) <= 5}
train_img_names = [x for x in opt.img_names if int(x[3:5])<=5]
Ntotal = len(train_img_names)
Ntrain = int(Ntotal * 0.8)
perm = np.random.permutation(Ntotal)

optimizer = torch.optim.SGD(rcnn.parameters(), lr=5e-4)

def load_data(Nimg, is_val):
    # loading process for training and testing might be different
    imgs = []
    img_info = []
    roi = []
    cls = []
    tbboxes = []    #tbbox : target bbox

    if not is_val:
        idx_l, idx_r = 0, Ntrain
    else:
        idx_l, idx_r = Ntrain, Ntotal
    gid = 0

    while 1:
        x = np.random.choice(range(idx_l, idx_r))
        img_name = train_img_names[perm[x]]
        fname = img_name[:-4]
        if not (fname in train_anno): continue
        info = train_anno[fname]
        bboxes = get_proposals(opt.proposal_path + 'train/' + fname + '.txt')
        bboxes = bboxes[:, :4]
        nroi = len(bboxes)
        gt_boxes = []

        for person in info:
            bbox = person['pos']
            if bbox[3] < 50: continue
            if person['occl'] == 1:
                # filter bboxes which are occluded more than 70%
                vbbox = person['posv']
                if isinstance(vbbox, int):
                    # it seems that sometimes 'posv' is mixed up with 'lock'
                    vbbox = person['lock']
                    assert isinstance(vbbox, list) and len(vbbox) == 4
                w1, h1, w2, h2 = bbox[2], bbox[3], vbbox[2], vbbox[3]
                if w2 * h2 / (w1 * h1) <= 0.3: continue
            bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # l, t, w, h ==> l, t, r, b
            gt_boxes.append(bbox)
        gt_boxes = np.array(gt_boxes)
        # maybe incorrect: images without positive examples are not included
        if len(gt_boxes) == 0: continue

        img, img_size = get_image(opt.image_path + img_name)

        rbboxes = rel_bbox(img_size, bboxes)
        ious = calc_ious(bboxes, gt_boxes)
        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)
        tbbox = bbox_transform(bboxes, gt_boxes[max_idx])

        pos_idx = []
        neg_idx = []

        for j in range(nroi):
            roi.append(rbboxes[j])
            tbboxes.append(tbbox[j])

            if max_ious[j] >= 0.5:
                pos_idx.append(gid)
                cls.append(1)
            else:
                neg_idx.append(gid)
                cls.append(0)
            gid += 1

        pos_idx = np.array(pos_idx)
        neg_idx = np.array(neg_idx)
        imgs.append(img)
        img_info.append({
            'img_size': img_size,
            'pos_idx': pos_idx,
            'neg_idx': neg_idx,
        })

        Nimg -= 1
        if Nimg == 0: break

    return np.array(imgs), img_info, np.array(roi), np.array(cls), np.array(tbboxes).astype(np.float32)

def train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=False):
    sc, r_bbox = rcnn(img, rois, ridx)
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)
    #print(loss.data.cpu().numpy())
    fl = loss.data.cpu().numpy()
    fl_sc = loss_sc.data.cpu().numpy()
    fl_loc = loss_loc.data.cpu().numpy()

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc

def train():
    print(f'=====================start training======================')
    I = 2
    B = 80
    POS = int(B * 0.25)
    NEG = B - POS

    is_val = False

    # if is_val:
        # rcnn.eval()
    # else:
        # rcnn.train()

    losses = []
    losses_sc = []
    losses_loc = []
    for idx in range(1000):
        train_imgs, infos, rois, gt_cls, gt_tbbox = load_data(I, is_val)
        img = Variable(torch.from_numpy(train_imgs), volatile=is_val).cuda()

        ridx = []
        glo_ids = []

        for j in range(I):
            info = infos[j]
            pos_idx = info['pos_idx']
            neg_idx = info['neg_idx']
            ids = []

            if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=POS)) #np.random.choice allows duplicate
            if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=NEG))
            if len(ids) == 0:
                continue
            ids = np.concatenate(ids, axis=0)
            glo_ids.extend(ids)
            ridx += [j] * ids.shape[0]

        if len(ridx) == 0:
            continue
        glo_ids = np.array(glo_ids)
        ridx = np.array(ridx)

        rois = rois[glo_ids]
        gt_cls = Variable(torch.from_numpy(gt_cls[glo_ids]).long(), volatile=is_val).cuda()
        gt_tbbox = Variable(torch.from_numpy(gt_tbbox[glo_ids]), volatile=is_val).cuda()

        loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

        if idx % 500 == 0:
            avg_loss = np.mean(losses)
            avg_loss_sc = np.mean(losses_sc)
            avg_loss_loc = np.mean(losses_loc)
            print(f'Iter {idx}: Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')


train()
torch.save(rcnn.state_dict(), opt.data_path + 'hao123.mdl')

