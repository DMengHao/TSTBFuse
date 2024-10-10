import kornia
import numpy as np
import torch
from skimage.io import imread
from torch import nn

from preprocessing import preprocessing

MSELoss = nn.MSELoss(reduction='mean')
Local_SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')
Global_SSIMLoss = kornia.losses.SSIMLoss(11, reduction='mean')
Gradient_loss = nn.L1Loss(reduction='mean')

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()
if __name__ == '__main__':
    ir_image = torch.tensor(imread('../ir.png').astype(np.float32)[None,None, :, :]/255.0)
    vi_image = np.transpose(imread('../vi.png').astype(np.float32), axes=(2, 0, 1)) / 255.0
    vi_image = torch.tensor(preprocessing.RGB_to_2Y(vi_image)[None,:,:,:])
    fusion_image = torch.tensor(imread('../Fusion_img.png').astype(np.float32)[None,None, :, :]/255.0)
    ir_fusion_mse_loss = MSELoss(ir_image,fusion_image)
    vi_fusion_mse_loss = MSELoss(vi_image,fusion_image)
    ir_fusion_local_ssim = Local_SSIMLoss(ir_image,fusion_image)
    vi_fusion_global_ssim = Local_SSIMLoss(vi_image,fusion_image)
    ir_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(ir_image), kornia.filters.SpatialGradient()(fusion_image))
    vi_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(vi_image), kornia.filters.SpatialGradient()(fusion_image))
    print('ir_fusion_mse_loss:', ir_fusion_mse_loss.item())
    print('vi_fusion_mse_loss:', vi_fusion_mse_loss.item())
    print('ir_fusion_local_ssim:', ir_fusion_local_ssim.item())
    print('vi_fusion_global_ssim:', vi_fusion_global_ssim.item())
    print('ir_fusion_gradient_loss:', ir_fusion_gradient_loss.item())
    print('vi_fusion_gradient_loss:', vi_fusion_gradient_loss.item())
