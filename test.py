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
opt = Config()
test_anno = { x : opt.annotation[x] for x in opt.annotation if int(x[3:5]) > 5 }
test_img_names = [x for x in opt.img_names if int(x[3:5]) > 5]
Ntest = len(test_img_names)

def load_data(idx):
    rois = []
    orig_rois = []

    img_name = test_img_names[idx]
    fname = img_name[:-4]
    bboxes = get_proposals(opt.proposal_path + 'test/' + fname + '.txt')
    bboxes = bboxes[:, :4]
    nroi = len(bboxes)
    img, img_size = get_image(opt.image_path + img_name)

    rbboxes = rel_bbox(img_size, bboxes)

    for j in range(nroi):
        rois.append(rbboxes[j])
        orig_rois.append(bboxes[j])

    test_img_info = {'img_size': img_size}
    return img, test_img_info, np.array(rois), np.array(orig_rois)

def test_image(img, img_size, rois, orig_rois):
    nroi = rois.shape[0]
    ridx = np.zeros(nroi).astype(int)
    img = img.cuda()
    sc, tbbox = rcnn(img, rois, ridx)
    sc = nn.functional.softmax(sc)
    sc = sc.data.cpu().numpy()
    tbbox = tbbox.data.cpu().numpy()
    bboxs = reg_to_bbox(img_size, tbbox, orig_rois)

    res_bbox = []
    res_cls = []

    for c in range(1, N_CLASS+1):
        c_sc = sc[:,c]
        c_bboxs = bboxs[:,c,:]

        boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.6)
        res_bbox.extend(boxes)
        res_cls.extend([c] * len(boxes))

    if len(res_cls) == 0:
        for c in range(1, N_CLASS+1):
            c_sc = sc[:,c]
            c_bboxs = bboxs[:,c,:]

            boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.3)
            res_bbox.extend(boxes)
            res_cls.extend([c] * len(boxes))
        res_bbox = res_bbox[:1]
        res_cls = res_cls[:1]

    #print(res_cls)

    return np.array(res_bbox), np.array(res_cls)

def test():
    bbox_preds = []
    bbox_cls = []

    for i in range(Ntest):
        bbox_preds.append(np.ndarray([0, 4]))
        bbox_cls.append(np.ndarray([0, 1]))

    with torch.no_grad():
        for i in trange(Ntest):
            test_img, info, rois, orig_rois = load_data(i)
            img = Variable(torch.from_numpy(test_img[np.newaxis,:]))

            img_size = info['img_size']

            res_bbox, res_cls = test_image(img, img_size, rois, orig_rois)
            bbox_preds[i] = res_bbox
            bbox_cls[i] = res_cls

    #evaluate.evaluate(bbox_preds, bbox_cls)

    print('Test complete')

rcnn = RCNN().cuda()
rcnn.load_state_dict(torch.load(opt.checkpoint_path + 'hao123.mdl'))
test()