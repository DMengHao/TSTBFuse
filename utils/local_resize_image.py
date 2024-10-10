import os

import cv2
import numpy as np
from PIL import Image
# %matplotlib inline
# TNO1
# pt1=(126, 286)
# pt2=(241, 106)
# width1=100
# hight1 = 100
# width2=100
# hight2=100
# TNO2
# pt1=(305, 357)
# pt2=(129, 453)
# width1=100
# hight1 = 100
# width2=100
# hight2=100
# TNO3
# pt1=(270, 87)
# pt2=(440, 370)
# width1=100
# hight1 = 100
# width2=100
# hight2=100
# TNO4
# pt1=(205, 205)
# pt2=(229, 5)
# width1=100
# hight1 = 100
# width2=100
# hight2=100
# TNO5
# pt1=(215, 91)
# pt2=(53, 151)
# width1=100
# hight1 = 100
# width2=100
# hight2=100
# RoadScene
# path = '../all data/result_RoadScene/'
# if not os.path.exists(path):
#     os.makedirs(path)
# RoadScene1
# pt1=(440, 115)
# pt2=(128, 16)
# width1=54
# hight1 = 39
# width2=54
# hight2=39
# RoadScene2
# pt1=(428, 215)
# pt2=(228, 120)
# width1=60
# hight1 = 37
# width2=60
# hight2=37
# RoadScene3
# pt1=(234, 156)
# pt2=(260, 53)
# width1=50
# hight1 = 30
# width2=50
# hight2=30
# RoadScene4
# pt1=(421, 201)
# pt2=(45, 31)
# width1=50
# hight1 = 35
# width2=50
# hight2=35
# MSRS
path = '../all data/result_MSRS/'
if not os.path.exists(path):
    os.makedirs(path)
# MSRS1
# pt1=(455, 230)
# pt2=(27, 283)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MSRS2
# pt1=(170, 220)
# pt2=(272, 272)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MSRS3
# pt1=(143, 106)
# pt2=(299, 405)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MSRS4
# pt1=(420, 200)
# pt2=(318, 91)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MRI_CT
# MRI_CT1
# pt1=(87, 15)
# pt2=(98, 171)
# width1=50
# hight1 = 50
# width2=50
# hight2=50
# MRI_CT2
pt1=(93, 91)
pt2=(152, 123)
width1=50
hight1 = 50
width2=50
hight2=50
# MRI_CT3
# pt1=(107, 29)
# pt2=(104, 132)
# width1=50
# hight1 = 50
# width2=50
# hight2=50
# MRI_CT4
# pt1=(100, 107)
# pt2=(95, 181)
# width1=50
# hight1 = 50
# width2=50
# hight2=50
def show_cvimg(im):
    return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

def stack_image(path:str):
    # 在指定的位置绘制正方形并在右下角放大该局部位置
    # pt:正方形位置,width : 正方形宽度 scale: 放大倍数
    image = cv2.imread(path)
    h, w, c = image.shape

    if w % 2 != 0:
        if h % 2 != 0:  # 宽高都是奇数
            image = cv2.resize(image, (w - 1, h - 1))
        else:  # 宽是奇数，高是偶数
            image = cv2.resize(image, (w - 1, h))
    elif h % 2 != 0:  # 高是奇数，宽是偶数
        image = cv2.resize(image, (w, h - 1))

    # 定义缩放比
    scale = h / w
    # patch1
    pt1_ = (pt1[0] + width1, pt1[1] + hight1)
    cv2.rectangle(image, pt1, pt1_, (0, 0, 255), 2)
    # patch2
    pt2_ = (pt2[0] + width2, pt2[1] + hight2)
    cv2.rectangle(image, pt2, pt2_, (0, 0, 255), 2)
    # 要放大的部分
    patch1_ = image[pt1[1] + 2:pt1[1] + hight1 - 2, pt1[0] + 2:pt1[0] + width1 -2, :]
    t1 = patch1_.copy()
    cv2.rectangle(t1, (0, 0), (t1.shape[1]-1, t1.shape[0]-1), (0, 0, 255), 1)
    t1 = cv2.resize(t1, (int(w / 2), int(h / 2)))


    patch2_ = image[pt2[1] + 2:pt2[1] + hight2 - 2, pt2[0] +2:pt2[0] + width2 - 2, :]
    t2 = patch2_.copy()
    cv2.rectangle(t2, (0, 0), (t2.shape[1] - 1, t2.shape[0] - 1), (0, 0, 255), 1)
    t2 = cv2.resize(t2, (int(w / 2), int(h / 2)))

    patch = np.hstack((t1, t2))
    image_stack = np.vstack((image, patch))
    return image_stack


if __name__ == '__main__':
    temp_list = []
    for root, dict, files in os.walk('../all data/MRI_CT2/'):
        for file in files:
            temp_list.append(os.path.join(root, file))
    for i in range(len(temp_list)):
        cv2.imwrite(os.path.join('../all data/result_MRI_CT/',os.path.basename(temp_list[i])), stack_image(temp_list[i]), [cv2.IMWRITE_PNG_COMPRESSION, 0])


# # 保存图片
# image_stack = stack_image()
# cv2.imwrite('image_stack.png', image_stack, [cv2.IMWRITE_PNG_COMPRESSION, 0])

