from time import time

from model import *
from pre_proc import *

N_CLASS = 1
opt = Config()
rcnn = RCNN().cuda()
print(rcnn)

with open(opt.log_path, 'w') as f:
    f.write('N_iter: %d\n' % opt.n_iter)
    f.write('Batch_size: %d\n' % opt.batch_size)
    f.write('Pos_ratio: %.2f\n' % opt.pos_ratio)
    f.write('Random_seed: %d\n\n' % opt.seed)
train_anno = {x: opt.annotation[x] for x in opt.annotation if int(x[3:5]) <= 5}
train_img_names = [x for x in opt.img_names if int(x[3:5]) <= 5]
Ntotal = len(train_img_names)
Ntrain = int(Ntotal * 0.9)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
perm = np.random.permutation(Ntotal)

optimizer = torch.optim.SGD(rcnn.parameters(), lr=5e-4)


def load_data(n_pos, n_neg, is_val=False):
    # loading process for training and testing might be different
    imgs = []
    img_info = []
    roi = []
    cls = []
    tbboxes = []  # tbbox : target bbox
    attns = []

    if not is_val:
        idx_l, idx_r = 0, Ntrain
    else:
        idx_l, idx_r = Ntrain, Ntotal
    gid = 0

    pos_cnt, neg_cnt, n_empty = 0, 0, 0
    while True:
        x = np.random.choice(range(idx_l, idx_r))
        img_name = train_img_names[perm[x]]
        fname = img_name[:-4]
        if not (fname in train_anno): continue
        info = train_anno[fname]
        bboxes = get_proposals(opt.proposal_path + fname + '.txt')
        bboxes = bboxes[:, :4]
        nroi = len(bboxes)

        gt_boxes, _ = get_gt_boxes(info)
        # optional: include at most 4 empty image
        if len(gt_boxes) == 0 and opt.include_empty_image and n_empty < 5:
            n_empty += 1
            gt_boxes = np.array([[.0, .0, .0, .0]])
        elif len(gt_boxes) == 0:
            continue

        img, img_size = get_image(opt.image_path + img_name)
        rbboxes = rel_bbox(img_size, bboxes)
        ious = calc_ious(bboxes, gt_boxes)
        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)
        tbbox = bbox_transform(bboxes, gt_boxes[max_idx])
        # compute ground truth attention weight
        attn = gt_attn(img_size, gt_boxes)

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
        attns.append(attn.unsqueeze(0))

        if pos_cnt >= n_pos and neg_cnt >= n_neg:
            break

    return torch.cat(imgs, dim=0), img_info, np.array(roi), np.array(cls), np.array(tbboxes).astype(
        np.float32), torch.cat(attns, dim=0)


def train_batch_rcnn(img, rois, ridx, gt_cls, gt_tbbox, gt_attns, is_val=False):
    sc, r_bbox, attns = rcnn(img, rois, ridx, 'train_rcnn')
    loss, loss_sc, loss_loc, loss_a = rcnn.calc_loss(sc, r_bbox, attns, gt_cls, gt_tbbox, gt_attns)
    # print(loss.data.cpu().numpy())
    fl = loss.data.cpu().numpy()
    fl_sc = loss_sc.data.cpu().numpy()
    fl_loc = loss_loc.data.cpu().numpy()
    fl_a = loss_a.data.cpu().numpy()

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc, fl_a


def train_batch_trans(img, rois, ridx, gt_cls, is_val=False):
    sc, trans = rcnn(img, rois, ridx, 'train_trans')
    loss, loss_sc, loss_trans = rcnn.calc_loss_trans(sc, trans, gt_cls)
    # print(loss.data.cpu().numpy())
    fl = loss.data.cpu().numpy()
    fl_sc = loss_sc.data.cpu().numpy()
    fl_trans = loss_trans.data.cpu().numpy()

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_trans


def valid(POS, NEG, type='train_rcnn'):
    print('Start to valid...')
    is_val = True
    rcnn.eval()
    losses, losses_sc, losses_loc, losses_a, losses_trans = [], [], [], [], []
    t0 = time()
    for idx in range(opt.valid_iter):
        valid_imgs, infos, rois, gt_cls, gt_tbbox, gt_attns = load_data(POS, NEG, is_val=True)
        img = valid_imgs.cuda()
        gt_attns = gt_attns.cuda()

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

        if type == 'train_rcnn':
            loss, loss_sc, loss_loc, loss_a = train_batch_rcnn(img, rois, ridx, gt_cls, gt_tbbox, gt_attns,
                                                               is_val=is_val)
            losses.append(loss)
            losses_sc.append(loss_sc)
            losses_loc.append(loss_loc)
            losses_a.append(loss_a)
        elif type == 'train_trans':
            loss, loss_sc, loss_trans = train_batch_trans(img, rois, ridx, gt_cls, is_val=is_val)
            losses.append(loss)
            losses_sc.append(loss_sc)
            losses_trans.append(loss_trans)
        else:
            raise ('In validation: type not supported')

    rcnn.train()
    if type == 'train_rcnn':
        return np.mean(losses), np.mean(losses_sc), np.mean(losses_loc), np.mean(losses_a), time() - t0
    elif type == 'train_trans':
        return np.mean(losses), np.mean(losses_sc), np.mean(losses_trans), time() - t0


def train(type='train_rcnn'):
    print(f'=====================start training======================')
    rcnn.train()

    B = opt.batch_size
    POS = int(B * opt.pos_ratio)
    NEG = B - POS
    is_val = False

    n_iter, print_every, save_every, checkpoint_path = opt.n_iter, opt.print_every, opt.save_every, opt.checkpoint_path
    if type == 'train_trans':
        n_iter, print_every, save_every, checkpoint_path = 30000, 200, 2000, opt.after_trans_path
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    losses, losses_sc, losses_loc, losses_a, losses_trans = [], [], [], [], []
    t0 = time()
    for idx in range(1, n_iter + 1):
        if idx == 45001:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
        train_imgs, infos, rois, gt_cls, gt_tbbox, gt_attns = load_data(POS, NEG)
        img = train_imgs.cuda()
        gt_attns = gt_attns.cuda()

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

        # shuffle in batch
        rand_idx = np.array(range(len(ridx)))
        np.random.shuffle(rand_idx)
        glo_ids = glo_ids[rand_idx]
        ridx = ridx[rand_idx]

        rois = rois[glo_ids]
        gt_cls = torch.LongTensor(gt_cls[glo_ids]).cuda()
        gt_tbbox = torch.FloatTensor(gt_tbbox[glo_ids]).cuda()
        if type == 'train_rcnn':
            loss, loss_sc, loss_loc, loss_a = train_batch_rcnn(img, rois, ridx, gt_cls, gt_tbbox, gt_attns,
                                                               is_val=is_val)
            losses.append(loss)
            losses_sc.append(loss_sc)
            losses_loc.append(loss_loc)
            losses_a.append(loss_a)
        elif type == 'train_trans':
            loss, loss_sc, loss_trans = train_batch_trans(img, rois, ridx, gt_cls, is_val=is_val)
            losses.append(loss)
            losses_sc.append(loss_sc)
            losses_trans.append(loss_trans)
        else:
            raise ('In training: type not supported')

        if idx % print_every == 0:
            if type=='train_rcnn':
                print_loss_rcnn(losses, losses_sc, losses_loc, losses_a, save_every, checkpoint_path, POS, NEG, t0, idx)
            elif type=='train_trans':
                print_loss_trans(losses, losses_sc, losses_trans, save_every, checkpoint_path, POS, NEG, t0, idx)
            t0 = time()
            losses, losses_sc, loss_loc, loss_a, loss_trans = [], [], [], [], []

def print_loss_rcnn(losses, losses_sc, losses_loc, losses_a, save_every, checkpoint_path, POS, NEG, t0, idx):
    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    avg_loss_a = np.mean(losses_a)
    t = time() - t0
    print(f'Iter {idx}: loss = {avg_loss:.4f}, loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}, loss_a = {avg_loss_a:.4f}, time = {t:.4f}')

    if idx % save_every == 0:  # save_every % print_every == 0
        l, l_sc, l_loc, l_a, t = valid(POS, NEG, 'train_rcnn')
        print(f'Iter {idx}: loss = {l:.4f}, loss_sc = {l_sc:.4f}, loss_loc = {l_loc:.4f}, loss_a = {l_a:.4f}, time = {t:.4f}')
        with open(checkpoint_path + 'log.txt', 'a+') as f:
            f.write(f'Train Iter {idx}: loss = {avg_loss:.4f}, loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}, loss_a = {avg_loss_a:.4f}\n')
            f.write(f'Valid Iter {idx}: loss = {l:.4f}, loss_sc = {l_sc:.4f}, loss_loc = {l_loc:.4f}, loss_a = {l_a:.4f}\n')
        print('Saving checkpoint...')
        torch.save(rcnn.state_dict(), checkpoint_path + 'iter_%d.mdl' % idx)


def print_loss_trans(losses, losses_sc, losses_trans, save_every, checkpoint_path, POS, NEG, t0, idx):
    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_trans = np.mean(losses_trans)
    t = time() - t0
    print(f'Iter {idx}: loss = {avg_loss:.4f}, loss_sc = {avg_loss_sc:.4f}, loss_trans = {avg_loss_trans:.4f}, time = {t:.4f}')

    if idx % save_every == 0:  # save_every % print_every == 0
        l, l_sc, l_trans, t = valid(POS, NEG, 'train_trans')
        print(f'Iter {idx}: loss = {l:.4f}, loss_sc = {l_sc:.4f}, loss_trans = {l_trans:.4f}, time = {t:.4f}')
        with open(checkpoint_path + 'log.txt', 'a+') as f:
            f.write(f'Train Iter {idx}: loss = {avg_loss:.4f}, loss_sc = {avg_loss_sc:.4f}, loss_trans = {avg_loss_trans:.4f}\n')
            f.write(f'Valid Iter {idx}: loss = {l:.4f}, loss_sc = {l_sc:.4f}, loss_trans = {l_trans:.4f}\n')
        print('Saving checkpoint...')
        torch.save(rcnn.state_dict(), checkpoint_path + 'iter_%d.mdl' % idx)


def prep_trans():
    print('Generating positive and negative centers...')
    t0 = time()

    # load model
    rcnn.load_state_dict(torch.load(opt.checkpoint_path + 'iter_90000.mdl'))

    rcnn.eval()
    # prepare R+ and R-
    idx_l, idx_r = 0, Ntotal
    pos_cnt, neg_cnt = 0, 0
    s1, tau1, s2, tau2 = opt.s1, opt.tau1, opt.s2, opt.tau2
    center_pos = np.zeros([512, 7, 7])
    center_neg = np.zeros([512, 7, 7])

    while True:
        x = np.random.choice(range(idx_l, idx_r))
        img_name = train_img_names[perm[x]]
        fname = img_name[:-4]
        if not (fname in train_anno): continue
        info = train_anno[fname]
        bboxes = get_proposals(opt.proposal_path + fname + '.txt')
        bboxes, scores = bboxes[:, :4], bboxes[:, 4]
        nroi = len(bboxes)

        gt_boxes, gt_occls = get_gt_boxes(info)
        if len(gt_boxes) == 0: continue

        img, img_size = get_image(opt.image_path + img_name)
        rbboxes = rel_bbox(img_size, bboxes)
        ious = calc_ious(bboxes, gt_boxes)
        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)

        pos_roi = []
        neg_roi = []
        #print(len(gt_occls), max_idx[:20])

        for j in range(nroi):
            if scores[j] >= s1 and max_ious[j] >= tau1 and not gt_occls[max_idx[j]]:
                pos_roi.append(rbboxes[j])
            if scores[j] < s2 and max_ious[j] < tau2:
                neg_roi.append(rbboxes[j])

        if len(pos_roi) == 0: continue
        pos_roi = np.array(pos_roi)
        neg_roi = np.array(neg_roi)
        np.random.shuffle(neg_roi)
        if len(neg_roi) > 50:
            neg_roi = neg_roi[:50, :]

        npos, nneg = len(pos_roi), len(neg_roi)
        #print(f'Pos Len = {len(pos_roi)}, Neg Len = {len(neg_roi)}')
        pos_cnt += npos
        neg_cnt += nneg

        img = img.unsqueeze(0).cuda()
        ridx = np.zeros(nroi).astype(int)
        pos_feat_sum, neg_feat_sum = rcnn.calc_center(img, pos_roi, neg_roi, ridx)
        center_pos += pos_feat_sum.data.cpu().numpy()
        center_neg += neg_feat_sum.data.cpu().numpy()

        if pos_cnt >= 2000 and neg_cnt >= 20000:
            break

    center_pos /= pos_cnt
    center_neg /= neg_cnt
    # print(center_pos.mean(), center_neg.mean(), center_pos.std(), center_neg.std())
    print(f"Center generated. Positive ROI samples: {pos_cnt}, Negative ROI samples: {neg_cnt}, Time: {time() - t0}")
    t0 = time()
    print('Start training transformation branch...')
    rcnn.train()
    rcnn.set_trans(torch.FloatTensor(center_pos).cuda(), torch.FloatTensor(center_neg).cuda())


if __name__ == '__main__':
    #train('train_rcnn')
    prep_trans()
    train('train_trans')
