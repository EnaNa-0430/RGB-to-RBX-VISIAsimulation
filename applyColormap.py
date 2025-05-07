import cv2
import numpy as np

img_name = 'face4'

# 读取灰度通道图
red_gray   = cv2.imread(img_name + 'RBX_Red.png',   cv2.IMREAD_GRAYSCALE)
brown_gray = cv2.imread(img_name + 'RBX_Brown.png', cv2.IMREAD_GRAYSCALE)
x_gray     = cv2.imread(img_name + 'RBX_X.png',     cv2.IMREAD_GRAYSCALE)

# 为每个通道分配伪彩色
red_color   = cv2.applyColorMap(red_gray,   cv2.COLORMAP_HOT)
brown_color = cv2.applyColorMap(brown_gray, cv2.COLORMAP_PINK)
x_color     = cv2.applyColorMap(x_gray,     cv2.COLORMAP_HOT)

# 保存彩色结果
cv2.imwrite(img_name + 'RBX_Red_color.png',   red_color)
cv2.imwrite(img_name + 'RBX_Brown_color.png', brown_color)
cv2.imwrite(img_name + 'RBX_X_color.png',     x_color)
