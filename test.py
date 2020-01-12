import numpy as np
from model import *
from utils import *
from tqdm import trange
from pre_proc import *
from vis_tool import *

N_CLASS = 1
fix = ''

opt = Config()
test_anno = {x: opt.annotation[x] for x in opt.annotation if int(x[3:5]) > 5}
test_img_names = [x for x in opt.img_names if int(x[3:5]) > 5]
Ntest = len(test_img_names)


def load_data(idx):
    rois = []
    orig_rois = []

    img_name = test_img_names[idx]
    fname = img_name[:-4]
    bboxes = get_proposals(opt.proposal_path + fname + '.txt')
    nroi = len(bboxes)
    img, img_size = get_image(opt.image_path + img_name)

    # bboxes = bboxes[:, :4]
    rbboxes = rel_bbox(img_size, bboxes)

    for j in range(nroi):
        rois.append(rbboxes[j])
        orig_rois.append(bboxes[j])

    test_img_info = {'img_size': img_size, 'fname': fname}
    return img, test_img_info, np.array(rois), np.array(orig_rois)


def test_image(img, img_size, rois, orig_rois):
    nroi = rois.shape[0]
    ridx = np.zeros(nroi).astype(int)
    img = img.cuda()
    sc, tbbox = rcnn(img, rois, ridx, 'test')
    # sc = nn.functional.softmax(sc, dim=1)
    sc = sc.data.cpu().numpy()
    tbbox = tbbox.data.cpu().numpy()
    bboxs = reg_to_bbox(img_size, tbbox, orig_rois)
    # rois = rois[:, np.newaxis, :]
    # rois = rois.repeat(N_CLASS + 1, axis=1)
    # bboxs = reg_to_bbox(img_size, rois, orig_rois)

    res_bbox = []
    res_score = []

    c = 1
    c_sc = sc[:, c]
    c_bboxs = bboxs[:, c, :]
    if fix == 'fix_bb' or fix == 'fix_all':
        c_bboxs = orig_rois[:, :-1]
    if fix == 'fix_cl' or fix == 'fix_all':
        c_sc = orig_rois[:, -1]

    boxes, scores = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.5, score_threshold=-1e10)
    res_bbox.extend(boxes)
    res_score.extend(scores)

    return np.array(res_bbox), np.array(res_score)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def output(preds):
    result_dir = 'caltech/data/res/'
    mkdir(result_dir)

    if fix == '':
        result_dir += opt.model_path.split('/')[2] + '/'
    else:
        result_dir += fix + '/'
    mkdir(result_dir)
    for i in range(6, 11):
        mkdir(result_dir + 'set' + '%02d' % i)

    res = dict()
    for name in preds:
        content = preds[name]
        name = name.split('_')
        fname = result_dir + name[0] + '/' + name[1] + '.txt'
        frame = int(name[2]) + 1
        if len(content) == 0: continue
        for box in content:
            # l t r b ==> l t w h
            box = [frame] + [round(x, 2) for x in box]
            if not fname in res:
                res[fname] = [box]
            else:
                res[fname] += [box]

    for fname in res:
        content = res[fname]
        content.sort()
        # print(content)
        content = [' '.join([str(y) for y in x]) for x in content]
        with open(fname, 'w') as f:
            f.write('\n'.join(content))


def test():
    bbox_preds = dict()

    with torch.no_grad():
        for i in trange(Ntest):
            test_img, info, rois, orig_rois = load_data(i)
            # img = Variable(torch.from_numpy(test_img[np.newaxis,:]))
            img = test_img.unsqueeze(0)
            img_size = info['img_size']

            res_bbox, res_score = test_image(img, img_size, rois, orig_rois)
            if len(res_bbox) == 0: continue

            res_bbox = np.concatenate([res_bbox, res_score.reshape(-1, 1)], axis=1)
            # res_bbox = np.concatenate([np.array(boxes), np.array(scores).reshape(-1,1)], axis=1)
            # print(res_bbox)
            # exit(0)
            res_bbox[:, 2] -= res_bbox[:, 0]
            res_bbox[:, 3] -= res_bbox[:, 1]
            bbox_preds[info['fname']] = res_bbox

    output(bbox_preds)
    print('Test complete')


if __name__ == '__main__':
    rcnn = RCNN().cuda()
    rcnn.load_state_dict(torch.load(opt.model_path))
    rcnn.eval()
    test()
