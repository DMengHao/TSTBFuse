import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from utils.Image_read_and_save import image_read_cv2
from scipy.signal import convolve2d

class Valuation(object):
    @classmethod
    def check(cls, Fusion_img, ir=None, vi=None):
        if ir is None or vi is None:
            assert type(Fusion_img) == np.ndarray, 'Fusion_img should be a numpy array'
            assert len(Fusion_img.shape) == 2, 'Fusion_img dimensions should be 2 dimensional'
        else:
            assert type(Fusion_img) == type(ir) == type(vi) == np.ndarray, 'Fusion_img and ir and vi should be the same type'
            assert Fusion_img.shape == ir.shape == vi.shape, 'Fusion_img and ir and vi should be the same shape'
            assert len(Fusion_img.shape) == 2, 'Fusion dimensions should be 2 dimensional'

    @classmethod
    def EN(cls, img):
        cls.check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def MI(cls, Fusion_img, ir, vi):
        cls.check(Fusion_img, ir, vi)
        F = Fusion_img.flatten()
        IR = ir.flatten()
        VI = vi.flatten()
        F_IR = mutual_info_score(F, IR)
        F_VI = mutual_info_score(F, VI)
        return F_IR + F_VI

    @classmethod
    def MSE(cls, Fusion_img, ir, vi):
        cls.check(Fusion_img, ir, vi)
        result1 = np.mean((ir - Fusion_img)**2)
        result2 = np.mean((vi - Fusion_img)**2)
        return (result1 + result2) / 2

    @classmethod
    def SSIM(cls, image_F, ir, vi):
        '''
        :param data_range is the range difference of your pixel values
        '''
        data_range = 255.0
        cls.check(image_F, ir, vi)
        return ssim(image_F, ir, data_range = data_range) + ssim(image_F, vi ,data_range = data_range)

    @classmethod
    def SD(cls, img):
        cls.check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.check(img)
        RF = np.sqrt(np.mean((img[1:,:]-img[:-1,:])**2))
        CF = np.sqrt(np.mean((img[:,1:]-img[:,:-1])**2))
        return np.sqrt(RF**2+CF**2)

    @classmethod
    def SCD(cls, image_F, ir, vi):
        cls.check(image_F, ir, vi)
        imgF_ir = image_F - ir
        a1  = np.sum((imgF_ir - np.mean(imgF_ir))*(vi - np.mean(vi)))
        b1 = np.sqrt(  np.sum( (imgF_ir - np.mean(imgF_ir) )**2)  * (np.sum( (vi-np.mean(vi))**2 ) ))
        corr1 = a1/b1
        imgF_vi = image_F - vi
        a2 = np.sum((imgF_vi - np.mean(imgF_vi))*(ir - np.mean(ir)))
        b2 = np.sqrt( np.sum( ( imgF_vi - np.mean(imgF_vi) )**2 ) * np.sum( (ir-np.mean(ir))**2 ) )
        corr2 = a2/b2
        return corr1 + corr2

    @classmethod
    def compare_viff(cls, ref, dist):  # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp
    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F) + cls.compare_viff(image_B, image_F)

    @classmethod
    def Qabf_getArray(cls, img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls, aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF, AAF, QgAF, QaAF, QAF = np.zeros_like(aA), np.zeros_like(aA), np.zeros_like(aA), np.zeros_like(
            aA), np.zeros_like(aA)
        GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF * QaAF
        return QAF
    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)
        # 计算QABF
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno
if __name__ == '__main__':
    Fusion_img = image_read_cv2("/home/dmh/New/NewFusion/1.png" ,'GRAY')
    IR = image_read_cv2("../vi.png", "GRAY")
    VI = image_read_cv2("../ir.png", "GRAY")
    EN = Valuation.EN(Fusion_img)
    MI = Valuation.MI(Fusion_img, IR, VI)
    MSE = Valuation.MSE(Fusion_img, IR, VI)
    SSIM = Valuation.SSIM(Fusion_img, IR, VI)
    SD = Valuation.SD(Fusion_img)
    SF = Valuation.SF(Fusion_img)
    SCD = Valuation.SCD(Fusion_img, IR, VI)
    Qabf = Valuation.Qabf(Fusion_img, IR, VI)

    metric_result = np.zeros((9))
    metric_result[0] = EN
    metric_result[1] = SD
    metric_result[2] = SF
    metric_result[3] = MI
    metric_result[4] = SCD
    metric_result[6] = Qabf
    metric_result[8] = MSE
    metric_result[7] = SSIM
    print("Testing of evaluation indicators:")
    print("=" * 85)
    print("\t\t\t EN\t\t SD\t\t SF\t\t MI\t\t SCD\t VIF\t Qabf\t SSIM\t MSE")
    print("NewFusion" '\t' + str(np.round(metric_result[0], 2)) + '\t' +
          str(np.round(metric_result[1], 2)) + '\t' +
          str(np.round(metric_result[2], 2)) + '\t'+
          str(np.round(metric_result[3], 2)) + '\t' +
          str(np.round(metric_result[4], 2)) + '\t' +
          str(np.round(metric_result[5], 2)) + '\t'+ '\t' +
          str(np.round(metric_result[6], 2)) + '\t' + '\t' +
          str(np.round(metric_result[7], 2)) + '\t' +
          str(np.round(metric_result[8], 2)) + '\t'
          )
    print("=" * 85)

