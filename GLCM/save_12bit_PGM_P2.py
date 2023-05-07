#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def save_pgm_p2(filename, data):
    max_value = 2**12 - 1  # 12-bit の最大値
    height, width = data.shape

    # P2 形式のヘッダーを書き込みます
    with open(filename, "w") as f:
        f.write("P2\n")
        f.write(f"{width} {height}\n")
        f.write(f"{max_value}\n")

        # 各ピクセルの値を書き込みます
        for i in range(height):
            for j in range(width):
                f.write(f"{data[i, j]} ")
                if j % 16 == 0:
                    f.write("\n")


if __name__ == '__main__':
    # 12-bit のデータを模倣します (shape: (height, width))
    data_12bit = np.random.randint(0, 2**12 - 1, (512, 512), dtype=np.uint16)

    # 12-bit グレースケールデータを P2 形式の PGM ファイルに保存します
    save_pgm_p2("12bit_grayscale_image_p2.pgm", data_12bit)
