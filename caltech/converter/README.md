Caltech Pedestrian Dataset Converter
============================

# Requirements

- OpenCV 3.0+ (with Python binding)
- Python 2.7+, 3.4+, 3.5+
- NumPy 1.10+
- SciPy 0.16+

# Caltech Pedestrian Dataset

```bash
$ bash download.sh  # 将数据下载到caltech/data文件夹下，然后解压
$ python convert_annotations.py
$ python convert_seqs.py  # 转成jpg图片，训练集10Hz采样，测试集1Hz采样
```

Each `.seq` movie is separated into `.png` images. Each image's filename is consisted of `{set**}_{V***}_{frame_num}.png`. According to [the official site](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/), `set06`~`set10` are for test dataset, while the rest are for training dataset.

(Number of objects: 346621)

# Draw Bounding Boxes

```
$ python test_plot_annotations.py
```
