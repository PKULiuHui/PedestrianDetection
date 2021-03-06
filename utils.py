import numpy as np
import torch


def rel_bbox(size, bbox):
    bbox = bbox.astype(np.float32)
    bbox[:, 0] /= size[0]
    bbox[:, 1] /= size[1]
    bbox[:, 2] /= size[0]
    bbox[:, 3] /= size[1]
    return bbox


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


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1e-10
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1e-10
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.array([targets_dx, targets_dy, targets_dw, targets_dh]).T
    return targets


def gt_attn(img_size, gt_rois, factor=16):
    attn = np.zeros(img_size)
    for roi in gt_rois:
        l, t, r, b = int(round(roi[0])), int(round(roi[1])), int(round(roi[2])), int(round(roi[3]))
        attn[l:r, t:b] = 1
    attn = attn[8:img_size[0]:factor, 8:img_size[1]:factor]
    return torch.FloatTensor(attn).transpose(0, 1)

def get_gt_boxes(info):
    gt_boxes = []
    gt_occls = []

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
        gt_occls.append(person['occl'])
    gt_boxes = np.array(gt_boxes)
    return gt_boxes, gt_occls


def reg_to_bbox(img_size, reg, box):
    img_width, img_height = img_size
    bbox_width = box[:, 2] - box[:, 0]
    bbox_height = box[:, 3] - box[:, 1]
    bbox_ctr_x = box[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = box[:, 1] + 0.5 * bbox_height

    bbox_width = bbox_width[:, np.newaxis]
    bbox_height = bbox_height[:, np.newaxis]
    bbox_ctr_x = bbox_ctr_x[:, np.newaxis]
    bbox_ctr_y = bbox_ctr_y[:, np.newaxis]

    out_ctr_x = reg[:, :, 0] * bbox_width + bbox_ctr_x
    out_ctr_y = reg[:, :, 1] * bbox_height + bbox_ctr_y

    out_width = bbox_width * np.exp(reg[:, :, 2])
    out_height = bbox_height * np.exp(reg[:, :, 3])

    return np.array([
        np.maximum(0, out_ctr_x - 0.5 * out_width),
        np.maximum(0, out_ctr_y - 0.5 * out_height),
        np.minimum(img_width, out_ctr_x + 0.5 * out_width),
        np.minimum(img_height, out_ctr_y + 0.5 * out_height)
    ]).transpose([1, 2, 0])


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
