import os

import cv2
import numpy as np

# 指定保存图像的文件夹和文件名
folder_path = 'D:/Thesis_source_code/NewFusion/Results/Roadsence'
file_name = 'fused_image.jpg'

# 构建完整的文件路径
full_path = os.path.join(folder_path, file_name)
print(full_path)
# 读取图像
# image1 = cv2.imread('/home/dmh/New/NewFusion/Datasets/Roadsence/ir/FLIR_00018.jpg')
# image2 = cv2.imread('/home/dmh/New/NewFusion/Datasets/Roadsence/vi/FLIR_00018.jpg')
image1 = cv2.imread('D:/Thesis_source_code/NewFusion/Datasets/Roadsence/ir/FLIR_00018.jpg')
image2 = cv2.imread('D:/Thesis_source_code/NewFusion/Datasets/Roadsence/vi/FLIR_00018.jpg')

# 转换为灰度图像（如果需要）
print(123)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# 调整图片大小
image1_gray = cv2.resize(image1_gray, (500, 500))
image2_gray = cv2.resize(image2_gray, (500, 500))
print(image1_gray,image2_gray)
# 归一化（如果已经是灰度图像且像素值在[0, 255]范围内，则此步骤可省略）
# image1_gray = image1_gray / 255.0
# image2_gray = image2_gray / 255.0

# 定义权重
weights = [0.5, 0.5]  # 示例权重，表示两张图像的重要性相同

# 融合图像
fused_image = (weights[0] * image1_gray.astype(np.float32) + weights[1] * image2_gray.astype(np.float32)).astype(np.uint8)

# 可视化结果
cv2.imshow('Fused Image', fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('D:/Thesis_source_code/NewFusion/Results/Roadsence/fused_image.jpg', fused_image)