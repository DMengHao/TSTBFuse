import os
import sys
import time
import datetime
import cv2
import kornia
from torch import nn
from torch.utils.data import DataLoader
import torch
from MyDataset import MyDataset
from Test.DenseBlock import DenseBlock as encoder1
from Test.Inception import Inception as encoder2
from Test.VIFEM import VIFEM as encoder3
from Test.Restormer_Decoder import Restormer_Decoder
import utils.Loss_function as loss_function
from utils.Draw_loss_curve import Draw_loss_curve
from utils.Loss_function import cc
lr = 1e-4
weight_decay = 0
optim_step = 1
optim_gamma = 0.5
epochs = 6
clip_grad_norm_value = 0.01
batch_size = 8
first_phase = 3
second_phase = 3
ir_grad_parameters = 4
vi_grad_parameters = 2
local_ssim = 7
ir_mse = 1
global_ssim = 6
vi_mse = 1
def train():
    dataset = MyDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = {'train': dataloader, }
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    prev_time = time.time()
    print("***********Dataloader Finished***********")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        Encoder1 = nn.DataParallel(encoder1()).to(device=device)
        Encoder2 = nn.DataParallel(encoder2()).to(device=device)
        Encoder3 = nn.DataParallel(encoder3()).to(device=device)
        Decoder = nn.DataParallel(Restormer_Decoder()).to(device=device)
    elif torch.cuda.device_count() == 1:
        Encoder1 = encoder1().to(device)
        Encoder2 = encoder2().to(device)
        Encoder3 = encoder3().to(device)
        Decoder = Restormer_Decoder().to(device)

    optimizer1 = torch.optim.AdamW(
        Encoder1.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = torch.optim.AdamW(
        Encoder2.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer3 = torch.optim.AdamW(
        Encoder3.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer4 = torch.optim.AdamW(
        Decoder.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

    MSELoss = loss_function.MSELoss
    Local_SSIMLoss = loss_function.Local_SSIMLoss
    Global_SSIMLoss = loss_function.Global_SSIMLoss
    Gradient_loss = loss_function.Gradient_loss

    mean_loss = []
    for epoch in range(epochs):
        Temp_loss = 0
        for index, (ir_img, vi_img) in enumerate(dataloader):

            ir_img, vi_img = ir_img.to(device), vi_img.to(device)
            Encoder1.train()
            Encoder2.train()
            Encoder3.train()
            Decoder.train()

            Encoder1.zero_grad()
            Encoder2.zero_grad()
            Encoder3.zero_grad()
            Decoder.zero_grad()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            if epoch <first_phase:
                ir_vi_feature = Encoder1(ir_img, vi_img)
                ir_feature = Encoder2(ir_img)
                vi_feature = Encoder3(vi_img)
                ir_image = Decoder(ir_img, ir_vi_feature, ir_feature,None)
                vi_image = Decoder(vi_img, ir_vi_feature, None,vi_feature)

                mse_loss_ir = local_ssim * Local_SSIMLoss(ir_img, ir_image) + ir_mse * MSELoss(ir_img, ir_image)
                mse_loss_vi = global_ssim * Global_SSIMLoss(vi_img, vi_image) + vi_mse * MSELoss(vi_img, vi_image)
                ir_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(ir_img),
                                                        kornia.filters.SpatialGradient()(ir_image))
                vi_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(vi_img),
                                                        kornia.filters.SpatialGradient()(vi_image))
                loss = (mse_loss_ir + mse_loss_vi +
                        ir_grad_parameters * ir_fusion_gradient_loss + vi_grad_parameters * vi_fusion_gradient_loss)
                loss.backward()
                Temp_loss = Temp_loss + loss
                nn.utils.clip_grad_norm_(
                    Encoder1.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder3.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()

            elif epoch>=second_phase:
                ir_vi_feature = Encoder1(ir_img, vi_img)
                ir_feature = Encoder2(ir_img)
                vi_feature = Encoder3(vi_img)
                fusion_feature, _ = Decoder(vi_img, ir_vi_feature, ir_feature, vi_feature)

                mse_loss_ir = local_ssim * Local_SSIMLoss(ir_img, fusion_feature) + ir_mse * MSELoss(ir_img, fusion_feature)
                mse_loss_vi = global_ssim * Global_SSIMLoss(vi_img, fusion_feature) + vi_mse * MSELoss(vi_img, fusion_feature)
                ir_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(ir_img),
                                                        kornia.filters.SpatialGradient()(fusion_feature))
                vi_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(vi_img),
                                                        kornia.filters.SpatialGradient()(fusion_feature))
                loss = ( mse_loss_ir + mse_loss_vi  +
                            5 * ir_fusion_gradient_loss + 5 * vi_fusion_gradient_loss)
                Temp_loss = Temp_loss + loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    Encoder1.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder2.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Encoder3.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()

            batches_done = epoch * len(loader['train']) + index
            batches_left = epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
                % (
                    epoch+1,
                    epochs,
                    index + 1,
                    len(loader['train']),
                    loss.item(),
                    time_left,
                )
            )
            sys.stdout.flush()

        scheduler1.step()
        scheduler2.step()

        if not epoch < first_phase:
            scheduler3.step()
            scheduler4.step()
        if optimizer1.param_groups[0]['lr'] <= 1e-6:
            optimizer1.param_groups[0]['lr'] = 1e-6
        if optimizer2.param_groups[0]['lr'] <= 1e-6:
            optimizer2.param_groups[0]['lr'] = 1e-6
        if optimizer3.param_groups[0]['lr'] <= 1e-6:
            optimizer3.param_groups[0]['lr'] = 1e-6
        if optimizer4.param_groups[0]['lr'] <= 1e-6:
            optimizer4.param_groups[0]['lr'] = 1e-6

        mean_loss.append(Temp_loss/(dataset.__len__()/batch_size))
        print(mean_loss)
        if True:
            checkpoint = {
                'Encoder1': Encoder1.state_dict(),
                'Encoder2': Encoder2.state_dict(),
                'Encoder3': Encoder3.state_dict(),
                'Decoder': Decoder.state_dict(),
            }
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            torch.save(checkpoint, os.path.join(f"./model/NewFusion_{epoch+1}_" + timestamp + '.pth'))
    Draw_loss_curve(epochs, Mean_Loss=torch.tensor(mean_loss).cpu(),run_time=timestamp)
if __name__ == '__main__':
    train()