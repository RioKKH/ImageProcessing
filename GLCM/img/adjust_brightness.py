#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage import data, img_as_float
import matplotlib.pyplot as plt

def adjust_brightness(image, gain, offset):
    adjusted_image = image * gain + offset
    return np.clip(adjusted_image, 0, 1)

def main(gain=1.2, offset=0.1):
    # サンプル画像を読み込む
    image = img_as_float(data.moon())

    # ゲインとオフセットを設定
    #gain = 1.2
    #offset = 0.1

    # adjust_brightness()関数を使用して輝度を調整
    adjusted_image = adjust_brightness(image, gain, offset)

    # 元の画像と調整後の画像を並べて表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2.imshow(adjusted_image, cmap=plt.cm.gray)
    ax2.set_title(f'Adjusted image (gain={gain}, offset={offset})')
    ax2.axis('off')

    plt.show()
