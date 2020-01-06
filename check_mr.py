# check ground truth and proposals miss rate

import os
import json
import numpy as np

gt_path = 'caltech/data/consistent_annotations.json'
pp_path = 'caltech/data/proposals'
our_path = 'caltech/data/res/vgg16_lmb20/'
fix_bbox_path = 'caltech/data/res/fix_bb/'
fix_cls_path = 'caltech/data/res/fix_cl/'


# bug, miss rate very high
def generate_gt_files():
    out_dir = 'caltech/data/res/gt/'
    if os.path.exists(out_dir):
        os.system('rm -rf %s' % out_dir)
    os.mkdir(out_dir)
    annotation = json.loads(open(gt_path, "rb").read())
    for k in annotation:
        sid = int(k[3:5])
        vid = k[6:10]
        fid = int(k[11:])
        if sid < 6:
            continue
        cur_dir = os.path.join(out_dir, k[:5])
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

        gt_boxes = []
        for person in annotation[k]:
            bbox = person['pos']
            if bbox[3] < 50:
                continue
            if person['occl'] == 1:
                # filter bboxes which are occluded more than 70%
                vbbox = person['posv']
                if isinstance(vbbox, int):
                    # it seems that sometimes 'posv' is mixed up with 'lock'
                    vbbox = person['lock']
                    assert isinstance(vbbox, list) and len(vbbox) == 4
                w1, h1, w2, h2 = bbox[2], bbox[3], vbbox[2], vbbox[3]
                if w2 * h2 / (w1 * h1) <= 0.3:
                    continue
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            gt_boxes.append(bbox)

        with open(os.path.join(cur_dir, '%s.txt' % vid), 'a+') as f:
            for b in gt_boxes:
                f.write('%d %.2f %.2f %.2f %.2f %.2f\n' % (fid + 1, b[0], b[1], b[2], b[3], 1.0))
            f.write('%d 0.0 0.0 0.0 0.0 %.2f\n' % (fid + 1, 1.0))


def generate_pp_files():
    out_dir = 'caltech/data/res/pp/'
    if os.path.exists(out_dir):
        os.system('rm -rf %s' % out_dir)
    os.mkdir(out_dir)
    for fn in sorted(os.listdir(pp_path), key=lambda x: int(x[11:-4])):
        k = fn[:-4]
        sid = int(k[3:5])
        vid = k[6:10]
        fid = int(k[11:])
        if sid < 6:
            continue
        cur_dir = os.path.join(out_dir, k[:5])
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        sc, bbox = [], []
        for line in open(os.path.join(pp_path, fn), 'r'):
            line = line.strip()
            if len(line) < 1:
                continue
            x = line.split('\t')
            l, t, r, d, s = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[-1])
            bbox.append([l, t, r, d])
            sc.append(s)
        sc, bbox = np.array(sc), np.array(bbox)
        bbox, sc = non_maximum_suppression(sc, bbox, iou_threshold=0.5, score_threshold=0.0)
        with open(os.path.join(cur_dir, '%s.txt' % vid), 'a+') as f:
            for i in range(len(bbox)):
                l, t, w, h, s = bbox[i][0], bbox[i][1], bbox[i][2] - bbox[i][0], bbox[i][3] - bbox[i][1], sc[i]
                f.write('%d %.2f %.2f %.2f %.2f %.2f\n' % (fid + 1, l, t, w, h, s))
    return


def non_maximum_suppression(sc, bboxs, iou_threshold=0.7, score_threshold=0.6):
    nroi = sc.shape[0]
    idx = np.argsort(sc)[::-1]
    rb = 0
    while rb < nroi and sc[idx[rb]] >= score_threshold:
        rb += 1
    if rb == 0:
        return [], []
    idx = idx[:rb]
    sc = sc[idx]
    bboxs = bboxs[idx, :]
    ious = calc_ious(bboxs, bboxs)

    res_box = []
    res_score = []
    for i in range(rb):
        if i == 0 or ious[i, :i].max() < iou_threshold:
            res_box.append(bboxs[i])
            res_score.append(sc[i])

    return res_box, res_score


def calc_ious(ex_rois, gt_rois):
    ex_area = (1. + ex_rois[:, 2] - ex_rois[:, 0]) * (1. + ex_rois[:, 3] - ex_rois[:, 1])
    gt_area = (1. + gt_rois[:, 2] - gt_rois[:, 0]) * (1. + gt_rois[:, 3] - gt_rois[:, 1])
    area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))

    lb = np.maximum(ex_rois[:, 0].reshape((-1, 1)), gt_rois[:, 0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:, 2].reshape((-1, 1)), gt_rois[:, 2].reshape((1, -1)))
    tb = np.maximum(ex_rois[:, 1].reshape((-1, 1)), gt_rois[:, 1].reshape((1, -1)))
    ub = np.minimum(ex_rois[:, 3].reshape((-1, 1)), gt_rois[:, 3].reshape((1, -1)))

    width = np.maximum(1. + rb - lb, 0.)
    height = np.maximum(1. + ub - tb, 0.)
    area_i = width * height
    area_u = area_sum - area_i
    ious = area_i / area_u
    return ious


def fix_bbox():



if __name__ == '__main__':
    # generate_gt_files()
    # generate_pp_files()
    fix_bbox()
