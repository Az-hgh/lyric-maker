# -*- coding: utf-8 -*-
"""用软件自带的 OCR 引擎实测：它是什么、认不认中文。"""
import os, sys, time

ROOT = r'D:\A\lyric-maker'
sys.path.insert(0, os.path.join(ROOT, '.deps'))
sys.path.append(os.path.join(ROOT, '.deps-ocr'))

TD = os.path.join(os.environ.get('TEMP', '.'), 'ocrdemo')

import numpy as np
import cv2

print('=== 引擎信息 ===')
import rapidocr_onnxruntime as _r
print('  rapidocr_onnxruntime 版本:', getattr(_r, '__version__', '未知'))

from rapidocr_onnxruntime import RapidOCR
t0 = time.time()
ocr = RapidOCR(intra_op_num_threads=4)
print('  初始化耗时: %.1fs' % (time.time() - t0))

print()
print('=== 实际识别 ===')
for name in sorted(os.listdir(TD)):
    if not name.lower().endswith('.png'):
        continue
    p = os.path.join(TD, name)
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), -1)
    t0 = time.time()
    res, _ = ocr(img)
    dt = time.time() - t0
    boxes = res or []
    print('  %s: %d 个文本框, %.2fs' % (name, len(boxes), dt))
    for box in boxes[:4]:
        txt = box[1]
        score = box[2]
        print('      %-30r 置信度 %.3f' % (txt[:28], score))
