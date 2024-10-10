import os
import cv2
import numpy as np
import torch
from Test.DenseBlock import DenseBlock as encoder1
from Test.Inception import Inception as encoder2
from Test.VIFEM import VIFEM as encoder3
from Test.Restormer_Decoder import Restormer_Decoder
from utils.Image_read_and_save import img_save, image_read_cv2
from utils.Valuation import Valuation

model_pth = './model/IVIF.pth'
def bgr_to_ycrcb(one):
    one = one.astype('float32')
    (B, G, R) = cv2.split(one)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, cv2.merge([Cr,Cb])
def ycrcb_to_bgr(one,Cr,Cb):
    one = one.astype('float32')
    Y = one
    Cr = Cr
    Cb = Cb
    B = (Cb - 0.5) * 1. / 0.564 + Y
    R = (Cr - 0.5) * 1. / 0.713 + Y
    G = 1. / 0.587 * (Y - 0.299 * R - 0.114 * B)
    return cv2.merge([B, G, R])
def val():
    for dataset_name in ['RoadScene','MSRS']:
        Model_Name = 'TSTBFuse'
        print("The test result of " + dataset_name + ' :')
        test_folder = os.path.join('test_images', dataset_name)
        test_out_folder = os.path.join('Results', dataset_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if torch.cuda.device_count() > 1:
        #     Encoder1 = nn.DataParallel(encoder1()).to(device=device)
        #     Encoder2 = nn.DataParallel(encoder2()).to(device=device)
        #     Encoder3 = nn.DataParallel(encoder3()).to(device=device)
        #     Decoder = nn.DataParallel(Restormer_Decoder()).to(device=device)
        # else:
        Encoder1 = encoder1().to(device)
        Encoder2 = encoder2().to(device)
        Encoder3 = encoder3().to(device)
        Decoder = Restormer_Decoder().to(device)

        Encoder1.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder1'])
        Encoder2.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder2'])
        Encoder3.load_state_dict(torch.load(model_pth, weights_only=True)['Encoder3'])
        Decoder.load_state_dict(torch.load(model_pth, weights_only=True)['Decoder'])
        Encoder1.eval()
        Encoder2.eval()
        Encoder3.eval()
        Decoder.eval()
        with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder, 'ir')):
                I_VIS, CBCR = bgr_to_ycrcb(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='RGB'))
                data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[None, None, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[None, None, ...] / 255.0
                ir_img, vi_img = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
                ir_img, vi_img = ir_img.to(device), vi_img.to(device)
                ir_vi_feature = Encoder1(ir_img, vi_img )
                ir_feature = Encoder2(ir_img)
                vi_feature = Encoder3(vi_img)
                fusion_feature, _ = Decoder(vi_img, ir_vi_feature, ir_feature, vi_feature)
                data_normalized = (fusion_feature - torch.min(fusion_feature)) / (torch.max(fusion_feature) - torch.min(fusion_feature))
                data_scaled = (data_normalized * 255).cpu().numpy()
                fi = np.squeeze(data_scaled).astype(np.uint8)

                fi = ycrcb_to_bgr(fi,CBCR[:,:,0:1].squeeze(-1),CBCR[:,:,1:2].squeeze(-1)).astype(int)
                img_save(fi.astype(np.uint8), img_name.split(sep='.')[0], test_out_folder)

        eval_folder = test_out_folder
        ori_img_folder = test_folder
        metric_result = np.zeros((9))
        for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
            metric_result += np.array([Valuation.EN(fi), Valuation.SD(fi)
                                          , Valuation.SF(fi), Valuation.MI(fi, ir, vi)
                                          , Valuation.SCD(fi, ir, vi), Valuation.VIFF(fi, ir, vi),
                                            Valuation.Qabf(fi,ir,vi), Valuation.MSE(fi, ir, vi),
                                       Valuation.SSIM(fi, ir, vi)])

        metric_result /= len(os.listdir(eval_folder))
        print("\t\t\t EN\t\t  SD\t SF\t     MI\t     SCD\t VIF\t Qabf\t MSE\t SSIM")
        print(Model_Name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
              + str(np.round(metric_result[1], 2)) + '\t'
              + str(np.round(metric_result[2], 2)) + '\t'
              + str(np.round(metric_result[3], 2)) + '\t'
              + str(np.round(metric_result[4], 2)) + '\t'
              + str(np.round(metric_result[5], 2)) + '\t'
              + str(np.round(metric_result[6], 2)) + '\t'
              + str(np.round(metric_result[7], 2)) + '\t'
              + str(np.round(metric_result[8], 2))
              )

if __name__ == '__main__':
    val()