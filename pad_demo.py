import numpy as np
from mmdet.datasets.pipelines import Pad, Compose
import cv2
import glob

if __name__ == '__main__':

    pad = Pad(size_divisor=32)
    compose = Compose(
        [dict(type='RandomFlip', flip_ratio=0.5),
         dict(
             type='AutoAugment',
             policies=[[{
                 'type':
                     'Resize',
                 'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                 'multiscale_mode':
                     'value',
                 'keep_ratio':
                     True
             }],
                 [{
                     'type': 'Resize',
                     'img_scale': [(400, 1333), (500, 1333),
                                   (600, 1333)],
                     'multiscale_mode': 'value',
                     'keep_ratio': True
                 }, {
                     'type': 'RandomCrop',
                     'crop_type': 'absolute_range',
                     'crop_size': (384, 600),
                     'allow_negative_crop': True
                 }, {
                     'type':
                         'Resize',
                     'img_scale': [(480, 1333), (512, 1333),
                                   (544, 1333), (576, 1333),
                                   (608, 1333), (640, 1333),
                                   (672, 1333), (704, 1333),
                                   (736, 1333), (768, 1333),
                                   (800, 1333)],
                     'multiscale_mode':
                         'value',
                     'override':
                         True,
                     'keep_ratio':
                         True
                 }]]),
            dict(type='Pad', size_divisor=32),
        ])
    for filename in glob.glob('val/*.png'):
        img = cv2.imread(filename)
        result = {'img': img}
        r = compose(result)
        print(filename)
        print(r['img'].shape)
