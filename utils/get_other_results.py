import os

import numpy as np

from utils.Image_read_and_save import image_read_cv2
from utils.Valuation import Valuation

eval_folder = '../Other methods/DATFuse/'


metric_result = np.zeros((9))
print('================================================================================')
# for dataset_name in ['TNO', 'MSRS', 'RoadScene']:
for dataset_name in ['Roadscene']:
    ori_img_folder = '../Other methods/DATFuse/'+dataset_name+'/'
    for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
        fuse_image = os.path.join(eval_folder, dataset_name+'/result/'  + img_name)
        ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
        # fi = image_read_cv2(os.path.join(fuse_image.replace('.png', '.bmp')), 'GRAY')
        fi = image_read_cv2(os.path.join(fuse_image), 'GRAY')
        metric_result += np.array([Valuation.EN(fi), Valuation.SD(fi)
                                      , Valuation.SF(fi), Valuation.MI(fi, ir, vi)
                                      , Valuation.SCD(fi, ir, vi), Valuation.VIFF(fi, ir, vi),
                                        Valuation.Qabf(fi,ir,vi), Valuation.MSE(fi, ir, vi),
                                   Valuation.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder+dataset_name+'/ir'))
    print("\t\t\t EN\t\t  SD\t SF\t     MI\t     SCD\t VIF\t Qabf\t MSE\t SSIM")
    print("Model_Name" + '\t' + str(np.round(metric_result[0], 2)) + '\t'
          + str(np.round(metric_result[1], 2)) + '\t'
          + str(np.round(metric_result[2], 2)) + '\t'
          + str(np.round(metric_result[3], 2)) + '\t'
          + str(np.round(metric_result[4], 2)) + '\t'
          + str(np.round(metric_result[5], 2)) + '\t'
          + str(np.round(metric_result[6], 2)) + '\t'
          + str(np.round(metric_result[7], 2)) + '\t'
          + str(np.round(metric_result[8], 2))
          )