import os
from tqdm import tqdm
import cv2

origion_path = '../test_images/MSRS/vi/'
save_path = '../test_images/MSRS_/vi/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
temp_list = []
for root, dirs,files in os.walk(origion_path):
    for file in files:
        temp_list.append(file)

for i in tqdm(range(len(temp_list))):
    img = cv2.imread(os.path.join(origion_path, temp_list[i]))
    grad_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_path, temp_list[i]), grad_img)

