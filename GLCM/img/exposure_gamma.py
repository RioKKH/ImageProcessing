#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float

def main(gamma=2.0, gain=1.0):
    # サンプル画像を読み込む
    image = img_as_float(data.moon())

    # ガンマ値とゲイン（オフセット）を設定
    #gamma = 2.0
    #gain = 1.0

    # exposure.adjust_gamma()を使用してガンマ補正を行う
    # 1. gamma
    # ガンマ値はガンマ補正の非線形変換の指数として使用されます。
    # ガンマ値が1より大きい場合、画像はより暗くなります。
    # ガンマ値が1より小さい場合、画像はより明るくなります。
    # ガンマ値が1の場合、画像は変更されません。  
    # 2. gain
    # ゲインは画像の輝度を調整するための乗数です。
    # ゲインが1より大きい場合には、画像全体の輝度が増加します。
    # ゲインが1より小さい場合には、画像全体の輝度が減少します。
    # ゲインが1の場合、輝度は変更されません。
    # adjusted_pixel_value = gain * (input_pixel_value ** gamma)
    adjusted_image = exposure.adjust_gamma(image, gamma, gain)

    # 元の画像と補正後の画像を並べて表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2.imshow(adjusted_image, cmap=plt.cm.gray)
    ax2.set_title(f'Adjusted image (gamma={gamma}, gain={gain})')
    ax2.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
