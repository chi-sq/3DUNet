# coding: utf-8
# @Author: Salt

'''
3d-unet train
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import config
from UNet_3D import UNet3D,UNet3D_BN
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import kidney_Loader
from utils import save_checkpoint, load_checkpoint
import numpy as np
import pandas as pd
from loss import FocalLoss, Dice_loss

torch.backends.cudnn.benchmark = True


# torch.set_default_tensor_type(torch.FloatTensor)

def train_fn(
        model, loader, opt_model, CE_loss, Focal_loss, lossopt=None,
):
    loop = tqdm(loader, leave=True)
    LOSS = 0
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE).type(torch.cuda.LongTensor)  # y->torch.Size([1, 1,  128,128,128)
        if x.shape != torch.Size([1, 1, 128, 128, 128]) or y.shape != torch.Size([1, 1, 128, 128, 128]):
            continue
        output = model(x)  # 没有softmax 在channel维度归一化(B,C,H,W,D) # output -> torch.Size([1,3,128,128,128])
        # In the segmentation, a value of 0 represents background, 1 represents kidney, and 2 represents tumor.

        if lossopt == 'Cross Entropy loss':  # -log(p_t)
            # y = y.squeeze(1)
            y = y.squeeze(1).permute(1, 2, 3, 0).view(-1)  
            output = output.permute(0, 2, 3, 4, 1).view(-1, 3)
            Loss = CE_loss(output, y)
        elif lossopt == 'Focal loss':  # -(1-p_t)^r * log(p_t)   # Loss = Focal_loss(output, y)
            y = y.squeeze(1).permute(1, 2, 3, 0).view(-1, 1)  
            output = output.permute(0, 2, 3, 4, 1).view(-1, 3)
            target, input = y, output
            logpt = F.log_softmax(input, -1)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())
            loss = -1 * (1 - pt) ** config.gamma * logpt
            Loss = loss.mean()
        elif lossopt == 'Dice loss':
            y = y.squeeze(1).permute(1, 2, 3, 0).squeeze(
                3)  
            output = output.squeeze(0).permute(1, 2, 3, 0)  # [H,W,D,C]
            f = nn.Softmax(dim=-1)
            output = torch.argmax(f(output), -1)  # [HWD]
            predictions = output
            predictions = predictions.type(torch.uint8)
            Loss = Dice_loss(predictions, y)
        elif lossopt == 'Mixed CE loss and Dice loss':
            y_1 = y.squeeze(1).permute(1, 2, 3, 0).view(-1)  
            output_1 = output.permute(0, 2, 3, 4, 1).view(-1, 3)

            y_2 = y.squeeze(1).permute(1, 2, 3, 0).squeeze(
                3)  # label [128,128,128,1]
            output_2 = output.squeeze(0).permute(1, 2, 3, 0)  # [H,W,D,C]
            f = nn.Softmax(dim=-1)
            output_2 = torch.argmax(f(output_2), -1)  # [HWD]
            predictions = output_2
            predictions = predictions.type(torch.uint8)

            lambda_ratio = 2
            CE = CE_loss(output_1, y_1)
            DICE = lambda_ratio * Dice_loss(predictions, y_2, lambda_tk=1, lambda_tu=1)  
            Loss = CE + DICE

        LOSS += Loss.item()

        opt_model.zero_grad()
        Loss.backward()
        opt_model.step()

        # if idx % 10 == 0:
        loop.set_postfix(
            Loss=LOSS / (idx + 1),
        )

    return LOSS / (idx + 1)


def val(
        loader, model, if_save_eva_index=None, csv_path='index/evaluation.csv'
):
    tk_dice_list = []
    tu_dice_list = []
    loop_val = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop_val):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE).type(torch.cuda.LongTensor)  # y->torch.Size([1, 1,  128,128,128)
        model.eval()
        if x.shape != torch.Size([1, 1, 128, 128, 128]) or y.shape != torch.Size([1, 1, 128, 128, 128]):
            continue
        with torch.no_grad():
            # In the segmentation, a value of 0 represents background, 1 represents kidney, and 2 represents tumor.
            y = y.squeeze(1).permute(1, 2, 3, 0).squeeze(3).to(
                'cpu').numpy()  
            output = model(x)  
            output = output.squeeze(0).permute(1, 2, 3, 0)  # [H,W,D,C]
            F = nn.Softmax(dim=-1)
            output = F(output).to('cpu').numpy()
            output = np.argmax(output, axis=-1)  # [HWD]
            predictions = output

            if not np.issubdtype(predictions.dtype, np.integer):
                predictions = np.round(predictions)
            predictions = predictions.astype(np.uint8)
        
            # Compute tumor+kidney Dice
            tk_pd = np.greater(predictions, 0)
            tk_gt = np.greater(y, 0)
            tk_dice = 2 * np.logical_and(tk_pd, tk_gt).sum() / (
                    tk_pd.sum() + tk_gt.sum()
            )

            # Compute tumor Dice
            tu_pd = np.greater(predictions, 1)
            tu_gt = np.greater(y, 1)
            tu_dice = 2 * np.logical_and(tu_pd, tu_gt).sum() / (
                    tu_pd.sum() + tu_gt.sum()
            )

            tk_dice_list.append(tk_dice)
            tu_dice_list.append(tu_dice)
        
            # 每次迭代打印dice系数的动态平均值
            loop_val.set_postfix(
                tk_dice=np.mean(tk_dice_list).item(),
                tu_dice=np.mean(tu_dice_list).item()
            )
    print(f'tk_dice = {np.mean(tk_dice_list)},tu_dice = {np.mean(tu_dice_list)}')
    if if_save_eva_index:
        data = pd.DataFrame({"tk_dice": tk_dice_list, "tu_dice": tu_dice_list})
        data.to_csv(csv_path, header=True)
        return np.mean(tk_dice_list)


def main():
    # model = UNet3D(in_channel=1, n_classes=3).to(config.DEVICE)
    model = UNet3D_BN(in_channel=1, n_classes=3,use_batch_norm=True).to(config.DEVICE)
    print(f'running on {config.DEVICE}')
    opt_model = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    CE_loss = nn.CrossEntropyLoss()
    Focal_loss = FocalLoss(gamma=2)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT, model, opt_model, config.LEARNING_RATE,
        )
    train_dataset = kidney_Loader(data_path=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_dataset = kidney_Loader(data_path=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    max_dice_coef = 0
    loss_list = []
    print("start to train")
    for epoch in range(config.NUM_EPOCHS):
        loss_per_epoch = train_fn(
            model, train_loader, opt_model, CE_loss, Focal_loss, lossopt=config.LOSS,
        )
        loss_list.append(loss_per_epoch)

        print(f"epoch{epoch + 1} done!!!")

        # if epoch % 5 == 0:
        print('start to validation')
        tk_dice_coef = val(val_loader, model, if_save_eva_index=True, csv_path=config.csv_path)

        if config.SAVE_MODEL and tk_dice_coef > max_dice_coef:
            save_checkpoint(model, opt_model, filename=config.CHECKPOINT_SAVE)
            max_dice_coef = tk_dice_coef

    data_loss = pd.DataFrame({"loss_per_epoch": loss_list})
    data_loss.to_csv('index/mixed_loss.csv', mode='a', header=True)
    print(f'max_dice_coef:{max_dice_coef}')


if __name__ == "__main__":
    main()
