#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/5/10 下午5:44
"""
import cv2
import numpy as np

for i in range(1, 21):
    img = cv2.imread(f"/home/msi/Pictures/s{i}.png")
    if img is not None:
        print(img.shape)
        img = img[350:1620, 150:2272 - 150]
        cv2.imwrite(f"../output/s{i}.png", img)
