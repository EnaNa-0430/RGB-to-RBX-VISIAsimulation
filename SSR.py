import cv2
import numpy as np


def SSR(img, sigma):
    img_log = np.log1p(np.array(img, dtype="float") / 255)  # log10(img + 1)
    img_fft = np.fft.fft2(img_log)  # Fourier
    G_recs = sigma // 2 + 1
    result = np.zeros_like(img_fft)
    rows, cols, deep = img_fft.shape
    for z in range(deep):
        for i in range(rows):
            for j in range(cols):
                for k in range(1, G_recs):
                    G = np.exp(-((np.log(k) - np.log(sigma)) ** 2) / (2 * np.log(2) ** 2))
                    result[i, j] += G * img_fft[i, j]
    img_ssr = np.real(np.fft.ifft2(result))  # inverse Fourier
    img_ssr = np.exp(img_ssr) - 1
    img_ssr = np.uint8(cv2.normalize(img_ssr, None, 0, 255, cv2.NORM_MINMAX))
    return img_ssr


if __name__ == "__main__":
    # 读取图像
    img_name = 'face1'
    img_path = img_name + '.jpg'
    image = cv2.imread(img_path)
    if image is None:
        print("Read error")
    else:
        # 调用 SSR 函数
        sigma = 10
        result = SSR(image, sigma)

        # 显示原始图像和处理后的图像
        cv2.imshow('Original Image', image)
        cv2.imshow('SSR Image', result)
        cv2.imwrite(img_name + '_SSR.png', result)

        # 等待按键关闭窗口
        cv2.waitKey(0)
        cv2.destroyAllWindows()
