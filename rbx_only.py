import numpy as np
from PIL import Image

# 1. 读取交叉偏振 RGB 图像
img_name = 'face4'
img = np.array(Image.open(img_name + '_SSR.png')).astype(np.float32)  # H×W×3  :contentReference[oaicite:6]{index=6}

# 2. 定义 3×3 变换矩阵 W2（示例占位，需替换为真实参数）
W2 = np.array([
    [0.562, -0.108, 0.202],  # 示例系数
    [-0.245, 0.756, -0.011],
    [0.123, -0.234, 0.567]
], dtype=np.float32)  # :contentReference[oaicite:7]{index=7}

# 3. 对每个像素应用线性变换
#    将 H×W×3 矩阵与 W2.T 相乘，结果为 H×W×3
rbx = img.dot(W2.T)  # H×W×3  :contentReference[oaicite:8]{index=8}

# 4. 分离通道
red_map = rbx[..., 0]  # 血红蛋白图
brown_map = rbx[..., 1]  # 黑色素图
x_map = rbx[..., 2]  # 残余通道


# 5. 归一化并保存结果
def save_gray(channel, name):
    chl = np.clip(channel, 0, None)
    chl = (chl / chl.max() * 255).astype(np.uint8)
    Image.fromarray(chl).save(f'{name}.png')


save_gray(red_map, img_name + 'RBX_Red')  # :contentReference[oaicite:9]{index=9}
save_gray(brown_map, img_name + 'RBX_Brown')  # :contentReference[oaicite:10]{index=10}
save_gray(x_map, img_name + 'RBX_X')  # :contentReference[oaicite:11]{index=11}
