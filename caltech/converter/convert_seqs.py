#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import cv2 as cv


def save_img(dname, fn, i, frame):
    cv.imwrite('{}/{}_{}_{}.jpg'.format(
        out_dir, os.path.basename(dname),
        os.path.basename(fn).split('.')[0], i), frame)

out_dir = '../data/images'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for idx, dname in enumerate(sorted(glob.glob('../data/set*'))):
    for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
        cap = cv.VideoCapture(fn)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx <= 5 and i % 3 == 0:
                save_img(dname, fn, i, frame)
            elif idx > 5 and i % 30 == 0:
                save_img(dname, fn, i, frame)
            i += 1
        print(fn)
