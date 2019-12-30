import numpy as np
from time import time
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from model import *
from utils import *
from pre_proc import *

N_CLASS = 1

rcnn = RCNN().cuda()
print(rcnn)

opt = Config()
if not os.path.exists(opt.checkpoint_path):
    os.mkdir(opt.checkpoint_path)
with open(opt.log_path, 'w') as f:
    f.write('N_iter: %d\n' % opt.n_iter)
    f.write('Batch_size: %d\n' % opt.batch_size)
    f.write('Pos_ratio: %.2f\n' % opt.pos_ratio)
    f.write('Random_seed: %d\n\n' % opt.seed)
train_anno = {x: opt.annotation[x] for x in opt.annotation if int(x[3:5]) <= 5}
train_img_names = [x for x in opt.img_names if int(x[3:5]) <= 5]
Ntotal = len(train_img_names)
Ntrain = int(Ntotal * 0.8)
np.random.seed(opt.seed)
perm = np.random.permutation(Ntotal)

optimizer = torch.optim.SGD(rcnn.parameters(), lr=5e-4)


# scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.1 ** (epoch // 45000)])

def load_data(n_pos, n_neg, is_val=False):
    # loading process for training and testing might be different
    imgs = []
    img_info = []
    roi = []
    cls = []
    tbboxes = []  # tbbox : target bbox

    if not is_val:
        idx_l, idx_r = 0, Ntrain
    else:
        idx_l, idx_r = Ntrain, Ntotal
    gid = 0

    pos_cnt, neg_cnt = 0, 0
    while 1:
        x = np.random.choice(range(idx_l, idx_r))
        img_name = train_img_names[perm[x]]
        fname = img_name[:-4]
        if not (fname in train_anno): continue
        info = train_anno[fname]
        bboxes = get_proposals(opt.proposal_path + fname + '.txt')
        bboxes = bboxes[:, :4]
        nroi = len(bboxes)
        gt_boxes = []
        gt_lbls = []

        for person in info:
            bbox = person['pos']
            # if person['lbl'] != 'person': continue
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
            gt_lbls.append(person['lbl'])
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

        pos_cnt += len(pos_idx)
        neg_cnt += len(neg_idx)

        pos_idx = np.array(pos_idx)
        neg_idx = np.array(neg_idx)
        imgs.append(img.unsqueeze(0))
        img_info.append({
            'img_size': img_size,
            'pos_idx': pos_idx,
            'neg_idx': neg_idx,
        })

        if pos_cnt >= n_pos and neg_cnt >= n_neg:
            break

    return torch.cat(imgs, dim=0), img_info, np.array(roi), np.array(cls), np.array(tbboxes).astype(np.float32)


def train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=False):
    sc, r_bbox = rcnn(img, rois, ridx)  # , gt_cls)
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)
    # print(loss.data.cpu().numpy())
    fl = loss.data.cpu().numpy()
    fl_sc = loss_sc.data.cpu().numpy()
    fl_loc = loss_loc.data.cpu().numpy()

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    return fl, fl_sc, fl_loc


def train():
    print(f'=====================start training======================')

    B = opt.batch_size
    POS = int(B * opt.pos_ratio)
    NEG = B - POS

    is_val = False

    losses = []
    losses_sc = []
    losses_loc = []
    t0 = time()
    for idx in range(1, opt.n_iter + 1):
        train_imgs, infos, rois, gt_cls, gt_tbbox = load_data(POS, NEG)
        img = train_imgs.cuda()

        pos_idx, neg_idx, pos_ridx, neg_ridx = [], [], [], []

        for j in range(len(train_imgs)):
            info = infos[j]
            pos_idx.extend(info['pos_idx'])
            pos_ridx.extend([j] * len(info['pos_idx']))
            neg_idx.extend(info['neg_idx'])
            neg_ridx.extend([j] * len(info['neg_idx']))

        pos_sample = np.random.choice(range(len(pos_idx)), POS, replace=False)
        neg_sample = np.random.choice(range(len(neg_idx)), NEG, replace=False)

        ridx, glo_ids = [], []
        for s in pos_sample:
            ridx.append(pos_ridx[s])
            glo_ids.append(pos_idx[s])
        for s in neg_sample:
            ridx.append(neg_ridx[s])
            glo_ids.append(neg_idx[s])

        glo_ids = np.array(glo_ids)
        ridx = np.array(ridx)

        rois = rois[glo_ids]
        gt_cls = torch.LongTensor(gt_cls[glo_ids]).cuda()
        gt_tbbox = torch.FloatTensor(gt_tbbox[glo_ids]).cuda()
        loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

        if idx % opt.print_every == 0:
            avg_loss = np.mean(losses)
            avg_loss_sc = np.mean(losses_sc)
            avg_loss_loc = np.mean(losses_loc)
            t = time() - t0
            print(
                f'Iter {idx}: Avg loss = {avg_loss:.4f}, loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}, time = {t:.4f}')

            if idx % opt.save_every == 0:  # save_every % print_every == 0
                l, l_sc, l_loc, t = valid(POS, NEG)
                print(f'Iter {idx}: Avg loss = {l:.4f}, loss_sc = {l_sc:.4f}, loss_loc = {l_loc:.4f}, time = {t:.4f}')
                with open(opt.log_path, 'a+') as f:
                    f.write(
                        f'Train Iter {idx}: Avg loss = {avg_loss:.4f}, loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}\n')
                    f.write(f'Valid Iter {idx}: Avg loss = {l:.4f}, loss_sc = {l_sc:.4f}, loss_loc = {l_loc:.4f}\n')
                print('Saving checkpoint...')
                torch.save(rcnn.state_dict(), opt.checkpoint_path + 'iter_%d.mdl' % idx)
            t0 = time()
            losses, losses_sc, loss_loc = [], [], []


def valid(POS, NEG):
    print('Start to valid...')
    is_val = True
    rcnn.eval()
    losses = []
    losses_sc = []
    losses_loc = []
    t0 = time()
    for idx in range(opt.valid_iter):
        valid_imgs, infos, rois, gt_cls, gt_tbbox = load_data(POS, NEG, is_val=True)
        img = valid_imgs.cuda()

        pos_idx, neg_idx, pos_ridx, neg_ridx = [], [], [], []

        for j in range(len(valid_imgs)):
            info = infos[j]
            pos_idx.extend(info['pos_idx'])
            pos_ridx.extend([j] * len(info['pos_idx']))
            neg_idx.extend(info['neg_idx'])
            neg_ridx.extend([j] * len(info['neg_idx']))

        pos_sample = np.random.choice(range(len(pos_idx)), POS, replace=False)
        neg_sample = np.random.choice(range(len(neg_idx)), NEG, replace=False)

        ridx, glo_ids = [], []
        for s in pos_sample:
            ridx.append(pos_ridx[s])
            glo_ids.append(pos_idx[s])
        for s in neg_sample:
            ridx.append(neg_ridx[s])
            glo_ids.append(neg_idx[s])

        glo_ids = np.array(glo_ids)
        ridx = np.array(ridx)

        rois = rois[glo_ids]
        gt_cls = torch.LongTensor(gt_cls[glo_ids]).cuda()
        gt_tbbox = torch.FloatTensor(gt_tbbox[glo_ids]).cuda()

        loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)
    rcnn.train()
    return np.mean(losses), np.mean(losses_sc), np.mean(loss_loc), time() - t0


if __name__ == '__main__':
    train()
