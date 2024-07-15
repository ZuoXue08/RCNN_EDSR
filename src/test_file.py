from PIL import Image
import numpy as np

# 图像路径
image_path = "/home/6c702main/EDSR-PyTorch-optuna/dataset/DIV2K/DIV2K_RCNN_train_LR_bicubic/X2/0001x2.png"

# 使用PIL打开图像
image = Image.open(image_path)

# 将图像转换为NumPy数组
image_array = np.array(image)

# 打印数组的形状
print(f"图像形状: {image_array.shape}")

