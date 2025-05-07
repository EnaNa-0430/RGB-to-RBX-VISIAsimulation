import cv2
import numpy as np


def MSR(img, scales):
    img_log = np.log1p(np.array(img, dtype="float") / 255)
    result = np.zeros_like(img_log)
    img_light = np.zeros_like(img_log)
    r, c, deep = img_log.shape
    for z in range(deep):
        for scale in scales:
            kernel_size = scale * 4 + 1
            # 高斯滤波器的大小，经验公式kernel_size = scale * 4 + 1
            sigma = scale
            img_smooth = cv2.GaussianBlur(img_log[:, :, z], (kernel_size, kernel_size), sigma)
            img_detail = img_log[:, :, z] - img_smooth
            result[:, :, z] += cv2.resize(img_detail, (c, r))
            img_light[:, :, z] += cv2.resize(img_smooth, (c, r))
    img_msr = np.exp(result+img_light) - 1
    img_msr = np.uint8(cv2.normalize(img_msr, None, 0, 255, cv2.NORM_MINMAX))
    return img_msr


if __name__ == "__main__":
    img_name = 'face4'
    img_path = img_name + '.png'
    image = cv2.imread(img_path)
    if image is None:
        print("Read error!")
    else:
        scales = [15, 80, 250]
        result = MSR(image, scales)

        cv2.imshow('Original Image', image)
        cv2.imshow('MSR Image', result)
        cv2.imwrite(img_name + '_MSR.png', result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()