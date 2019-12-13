import numpy as np
import json
from PIL import Image
import tqdm
from tqdm import trange
from utils import *
import os
import random
from vis_tool import vis_bbox
import matplotlib.pyplot as plt

### Train
class Config:
    data_path = './caltech/data/'
    image_path = './caltech/data/images/'
    proposal_path = './caltech/data/proposals/'

    annotation = json.loads(open(data_path + 'consistent_annotations.json', "rb").read())
    img_names = os.listdir(image_path)
    print('In total there are {} annotations'.format(len(annotation)))

def get_proposals(fname):
    file = open(fname, 'rb')
    proposals = str(file.read().strip())[2:-1].split('\\n')
    file.close()
    for i, x in enumerate(proposals):
        proposals[i] = np.array([float(y) for y in x.split('\\t')])
    return np.array(proposals)

def get_image(fname):
    img = Image.open(fname)
    img_size = img.size
    # print(img_size)
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)
    return np.transpose(img, [2, 0, 1]), img_size

'''
train_imgs = np.array(train_imgs)
train_img_info = np.array(train_img_info)
train_roi = np.array(train_roi)
train_cls = np.array(train_cls)
train_tbbox = np.array(train_tbbox).astype(np.float32)


### Test

test_data = [ [x] + annotation[x] for x in annotation if int(x[3:5]) > 5]

test_imgs = []
test_img_info = []
test_roi = []
test_orig_roi = []

print(len(test_data), test_data[1])
N_test = len(test_data)
for i in trange(N_test):
    info = test_data[i]
    img_name = info[0] + '.jpg'
    bboxes = get_proposals(proposal_path + 'test/' + info[0] + '.txt')
    bboxes = bboxes[:, :4]
    info = info[1:]
    nobj = len(info)
    nroi = len(bboxes)
    gt_boxes = []
    for person in info:
        bbox = person['pos']
        if bbox[3] < 50: continue
        #TODO: add other criterions
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # l, t, w, h ==> l, t, r, b
        gt_boxes.append(bbox)
    gt_boxes = np.array(gt_boxes)
    if len(gt_boxes) == 0: continue

    img, img_size = get_image(img_name)

    rbboxes = rel_bbox(img_size, bboxes)
    idxs = []

    for j in range(nroi):
        gid = len(test_roi)
        test_roi.append(rbboxes[j])
        test_orig_roi.append(bboxes[j])
        idxs.append(gid)

    idxs = np.array(idxs)
    test_imgs.append(img)
    test_img_info.append({
        'img_size': img_size,
        'idxs': idxs
    })
    # print(len(idxs))

test_imgs = np.array(test_imgs)
test_img_info = np.array(test_img_info)
test_roi = np.array(test_roi)
test_orig_roi = np.array(test_orig_roi)

print(test_imgs.shape)
print(test_roi.shape)
print(test_orig_roi.shape)

np.savez(open(data_path + 'test.npz', 'wb'),
         test_imgs=test_imgs, test_img_info=test_img_info, test_roi=test_roi, test_orig_roi=test_orig_roi)
'''