import torch
import torch.nn as nn
import torch.optim as optim
import config
from UNet_3D import UNet3D
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import kidney_Loader
from utils import save_checkpoint, load_checkpoint
import numpy as np
import pandas as pd

torch.backends.cudnn.benchmark = True


def val(loader, model, if_save_eva_index=None, csv_path='index/evaluation.csv'):
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
                'cpu').numpy()  # [1,128,128,128] # y value is 0,1,2 代表对应的类别的索引 label [128,128,128,1]
            # print("y.max(),y.min()",y.max(),y.min())
            output = model(x)  # 没有softmax 在channel维度归一化(B,C,H,W,D)
            output = output.squeeze(0).permute(1, 2, 3, 0)  # [H,W,D,C]
            F = nn.Softmax(dim=-1)
            output = F(output).to('cpu').numpy()
            output = np.argmax(output, axis=-1)  # [HWD]
            # max,min = output.max(),output.min()
            # print(max, min)
            predictions = output

            if not np.issubdtype(predictions.dtype, np.integer):
                predictions = np.round(predictions)
            predictions = predictions.astype(np.uint8)

            # print("predictions.max(),predictions.min()",predictions.max(),predictions.min())
            # Compute tumor+kidney Dice
            tk_pd = np.greater(predictions, 0)
            tk_gt = np.greater(y, 0)
            # tk_pd = (predictions==1)
            # tk_gt = (y == 1)
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
    print('validation ending--')
    print(f'tk_dice = {np.mean(tk_dice_list)},tu_dice = {np.mean(tu_dice_list)}')
    if if_save_eva_index:
        data = pd.DataFrame({"tk_dice": tk_dice_list, "tu_dice": tu_dice_list})
        data.to_csv(csv_path, header=True)


def main():
    print('start to validation')
    val_dataset = kidney_Loader(data_path=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet3D(in_channel=1, n_classes=3).to(config.DEVICE)
    opt_model = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT, model, opt_model, config.LEARNING_RATE,
        )
    val(val_loader, model, if_save_eva_index=False, csv_path='index/evaluation.csv')


if __name__ == "__main__":
    main()
