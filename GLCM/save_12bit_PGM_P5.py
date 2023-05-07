#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage import io

# 12-bit のデータを模倣します (shape: (height, width))
data_12bit = np.random.randint(0, 2**12 - 1, (512, 512), dtype=np.uint16)

# 12-bit のデータを 16-bit に変換します
data_16bit = data_12bit << 4

# 16-bit の PGM ファイルにデータを保存します
io.imsave("12bit_grayscale_image.pgm", data_16bit)
