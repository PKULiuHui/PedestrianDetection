import numpy as np
import json
from PIL import Image
import tqdm
from tqdm import trange
from utils import *
import os
import random
# from vis_tool import vis_bbox
import matplotlib.pyplot as plt

### Train
class Config:
    data_path = './caltech/data/'
    image_path = './caltech/data/images/'
    proposal_path = './caltech/data/proposals/'
    checkpoint_path = './caltech/ckpt20/'
    log_path = checkpoint_path + 'log.txt'
    print_every = 500
    save_every = 10000
    seed = 123
    batch_size = 160
    pos_ratio = 0.25
    n_iter = 90000
    valid_iter = 100

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

from torchvision import transforms
Transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_image(fname):
    img = Image.open(fname)
    img_size = img.size
    # print(img_size)
    # img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    # img = np.transpose(img, [2, 0, 1])
    img = Transform(img)
    return img, img_size
