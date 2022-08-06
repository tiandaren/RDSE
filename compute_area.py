import argparse
#import paddle
import matplotlib.pyplot as plt
#import paddle.vision.transforms.functional as F
from PIL import Image
import os.path
import xml.etree.ElementTree as ET
from model import *
from data import *
from compute_utils import *
def test_simple(im0,X,Y,c,x1,y1,x2,y2,label):


    # image_path = p
    encoder='resnet'
    wrap_mode='border'
    checkpoint_path='weights/city2eigen_resnet.pdparams'
    input_height=256
    input_width=512

    if c==0:
        class_names = 'D00'
    elif c==2:
        class_names='D10'
    elif c==4:
        class_names = 'D20'
    elif c==5:
        class_names='D40'
    # input_image = np.array(Image.open(p).convert('RGB'))
    input_image=im0
    original_height, original_width, num_channels = input_image.shape
    input_image = F.resize(input_image, [input_height, input_width], interpolation='area')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    model = MonodepthModel(encoder)

    if checkpoint_path.endswith('.h5'):
        load_tensorflow_weight(model, checkpoint_path)
    elif checkpoint_path.endswith('.pdparams'):
        model.load_dict(paddle.load(checkpoint_path))

    disp, _ = model(paddle.to_tensor(input_images, dtype=paddle.float32).transpose((0, 3, 1, 2)))
    disp_pp = post_process_disparity(disp[0].squeeze().numpy()).astype(np.float32)

    #output_directory = os.path.dirname(image_path)
    #output_name = os.path.splitext(os.path.basename(image_path))[0]

    disp_to_img = F.resize(disp_pp.squeeze(), [original_height, original_width], interpolation='area')
    #disp_to_img 就是每个像素的深度估计值 是一个array 具体取一个值是 disp_to_img[y][x]






    distance1 = disp_to_img[y1 - 1][int((x1 + x2) / 2) - 1] * 10000
    distance1 = 700 - distance1
    distance2 = disp_to_img[y2 - 1][int((x1 + x2) / 2) - 1] * 10000
    distance2 = 700 - distance2
    h = distance1 - distance2
    if distance2 < 150:
        h *= 0.25
    elif distance2 < 170:
        h *= 0.4
    elif distance2 < 200:
        h *= 0.45
    elif distance2 < 250:
        h *= 0.5
    elif distance2 < 300:
        h *= 0.52
    elif distance2 < 350:
        h *= 0.54
    elif distance2 < 400:
        h *= 0.6
    elif distance2 < 450:
        h *= 0.72
    elif distance2 < 500:
        h *= 1.05
    elif distance2 < 550:
        h *= 1.07
    elif distance2 < 600:
        h *= 1.35
    elif distance2 < 700:
        h *= 1.45
    elif distance2 < 750:
        h *= 1.6
    elif distance2 < 800:
        h *= 2
    else:
        h *= 2.5

    if distance1 < 150:
        h *= 0.25
    elif distance1 < 170:
        h *= 0.4
    elif distance1 < 200:
        h *= 0.45
    elif distance1 < 250:
        h *= 0.5
    elif distance1 < 300:
        h *= 0.52
    elif distance1 < 350:
        h *= 0.54
    elif distance1 < 400:
        h *= 0.6
    elif distance1 < 450:
        h *= 0.72
    elif distance1 < 500:
        h *= 1.05
    elif distance1 < 550:
        h *= 1.07
    elif distance1 < 600:
        h *= 1.35
    elif distance1 < 700:
        h *= 1.45
    elif distance1 < 750:
        h *= 1.6
    elif distance1 < 800:
        h *= 2
    else:
        h *= 2.5

    h *= 5

    w = h / (y2 - y1) * (x2 - x1)
    if distance2 < 150:
        w *= 0.5
    elif distance2 < 200:
        w *= 0.3
    elif distance2 < 400:
        w *= 0.33
    elif distance2 < 600:
        w *= 0.25
    # 增加一定约束
    h = abs(h)
    w = abs(w)

    compensate = 0.75
    if str(class_names) == 'D00':
        if h < w:
            h, w = w, h
        if h < 50:
            h += (50 - h) * compensate
        if h > 200:
            h -= (h - 200) * compensate
        if w < 20:
            w += (20 - w) * compensate
        if w > 50:
            w -= (w - 50) * compensate

        h/=100
        h = round(h,2)
        label += " length="+str(h)
        return label
    # 补偿机制
    if str(class_names) == 'D10':
        if w < h:
            h, w = w, h
        if w < 50:
            w += (50 - w) * compensate
        if w > 200:
            w -= (w - 200) * compensate
        if h < 20:
            h += (20 - h) * compensate
        if h > 50:
            h -= (h - 50) * compensate
        w/=100
        w = round(w,2)
        label += " width=" + str(w)
        return label
    if str(class_names) == 'D20':
        if w < 50:
            w += (50 - w) * compensate
        if w > 150:
            w -= (w - 150) * compensate
        if h < 50:
            h += (50 - h) * compensate
        if h > 150:
            h -= (h - 150) * compensate
        h/=100
        h = round(h,2)
        w /= 100
        w = round(w, 2)
        area = round(w*h, 2)
        label += " area=" + str(area)
        return label
    if str(class_names) == 'D40':
        if w < 20:
            w += (20 - w) * compensate
        if w > 60:
            w -= (w - 60) * compensate
        if h < 20:
            h += (20 - h) * compensate
        if h > 60:
            h -= (h - 60) * compensate
        h/=100
        w /= 100

        area = round(w*h, 2)

        label += " area=" + str(area)
        return label


