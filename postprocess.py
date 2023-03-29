import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import config
from UNet_3D import UNet3D
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint
import nibabel as nib
from src.resize import resize_volume
import numpy as np
import pandas as pd
import os
torch.backends.cudnn.benchmark = True

def crop(image):  
    D, H, W = image.shape  # (611,512,512) 轴 冠 shi
    for k in range(D - 1):
        if image[k, :, :].max() > 0:
            top = k
            break
    for k in range(D - 1, -1, -1):
        if image[k, :, :].max() > 0:
            bottom = k
            break
    for i in range(H - 1):
        if image[:, i, :].max() > 0:
            left = i
            break
    for i in range(H - 1, -1, -1):
        if image[:, i, :].max() > 0:
            right = i
            break
    for j in range(W - 1):
        if image[:, :, j].max() > 0:
            forward = j
            break
    for j in range(W - 1, -1, -1):
        if image[:, :, j].max() > 0:
            backward = j
            break
    return top, bottom, left, right, forward, backward

def normalize(volume):
    """Normalize the volume"""
    # set different HU value according to ROI
    min = -512
    max = 512
    # Clip at max and min values if specified
    volume[volume < min] = min
    volume[volume > max] = max
    # normalize to [0,1]
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

# for one sample 1.ROI crop 2.window clip (normalize) 3.resize 4.totensor --> prediction = model(x) [1,1,128,128,128]
# 1.to numpy 2.resize inverse 3.unormalize 4.save to nii
def applicator(image_path=None): #image_path is case_00xxx
    print('start to predict')
    model = UNet3D(in_channel=1, n_classes=3).to(config.DEVICE)
    opt_model = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT, model, opt_model, config.LEARNING_RATE,
        )
    #
    image = nib.load(os.path.join(image_path, 'imaging.nii.gz'))
    seg_true = nib.load(os.path.join(image_path, 'segmentation.nii.gz'))
    affine = image.affine
    # check orientation
    print(f'Input Image Orientation {nib.aff2axcodes(image.affine)}')
    image = image.get_fdata()
    origin_depth,origin_width,origin_height = image.shape
    seg_true = seg_true.get_fdata()
    # crop
    top, bottom, left, right, forward, backward = crop(seg_true)
    top, bottom, left, right, forward, backward = top - 2, bottom + 2, left - 10, right + 10, forward - 10, backward + 10  # 留出bbox和感兴趣区域的宽度
    image = image[top:bottom, left:right, forward:backward]
    image_crop = image
    seg_true = seg_true[top:bottom, left:right, forward:backward]
    # resize volume to 128*128*128
    image = resize_volume(image)  # 128 128 128
    image = normalize(image)
    # totensor
    transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    image = transforms_(image).unsqueeze(0).unsqueeze(0) # 增加channel 和 batch
    image = image.to(config.DEVICE)
    seg_predict = model(image)# [1, 3, 128, 128, 128]
    seg_predict = seg_predict.squeeze(0).permute(1, 2, 3, 0) #[128,128,128,3]
    seg_predict = nn.Softmax(dim=-1)(seg_predict)
    print(seg_predict.shape)
    seg_predict = torch.argmax(seg_predict,dim=-1).to('cpu').numpy()  # [128,128,128]
    max = seg_predict.max()
    print(max)
    seg_predict = resize_volume(seg_predict,desired_depth=bottom-top,desired_height=right-left,desired_width=backward-forward)
    seg = nib.Nifti1Image(seg_predict.astype(np.int8), affine)
    image_crop = nib.Nifti1Image(image_crop, affine)
    seg_true = nib.Nifti1Image(seg_true.astype(np.int8), affine)
    print("===> saving image")
    nib.save(seg,'./seg_predict.nii.gz')
    nib.save(image_crop,'./image_crop.nii.gz')
    nib.save(seg_true,'./seg_true_crop.nii.gz')
    print('generated seg_prediction')

if __name__ == "__main__":
    applicator(image_path="data/val/case_00171")





