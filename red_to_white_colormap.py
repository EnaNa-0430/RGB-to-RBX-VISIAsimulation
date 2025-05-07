import cv2
import numpy as np


def create_red_to_white_colormap():
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        j = i#255 - i
        # 线性插值从鲜红色到白色
        blue = int(j) * 1
        green = int(j) * 1
        red = (255 - int((255 - j) * 0)) * 1
        colormap[i, 0] = [blue, green, red]
    return colormap


# 读取灰度图像
image = cv2.imread('face4RBX_Red.png', cv2.IMREAD_GRAYSCALE)

# 创建自定义颜色映射表
custom_colormap = create_red_to_white_colormap()

# 应用自定义颜色映射
colored_image = cv2.applyColorMap(image, custom_colormap)

# 显示原始图像和应用颜色映射后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Colored Image', colored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

