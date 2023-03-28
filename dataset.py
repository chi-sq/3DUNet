# coding: utf-8
# @Author: Salt

import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from src.resize import resize_volume
from torchvision import transforms
import torch


class kidney_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的case
        self.data_path = data_path
        self.imgs_path = os.listdir(self.data_path)
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

    def augment(self, image):
        pass

    def crop(self, image):  # image refers to segmentaiotn
        D, H, W = image.shape  # (611,512,512) 轴 冠 shi
        # H维度crop
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

    # 把CT的HU值归一化到0-1
    def normalize(self, volume):
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

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        image = nib.load(os.path.join(self.data_path, image_path, 'imaging.nii.gz'))
        seg = nib.load(os.path.join(self.data_path, image_path, 'segmentation.nii.gz'))
        # get_fdata()方法转成ndarray
        image = image.get_fdata()  # 611 512 512
        seg = seg.get_fdata()
        # crop automatically
        top, bottom, left, right, forward, backward = self.crop(seg)
        top, bottom, left, right, forward, backward = top - 2, bottom + 2, left - 10, right + 10, forward - 10, backward + 10  # 留出bbox和感兴趣区域的宽度
        image = image[top:bottom, left:right, forward:backward]
        seg = seg[top:bottom, left:right, forward:backward]
        # resize volume to 128*128*128
        image = resize_volume(image)  # 128 128 128
        seg = resize_volume(seg)
        # normalize
        image = self.normalize(image)
        # seg = self.normalize(seg) #不需要normalize
        # 转换成tensor 然后加入channel维度
        image = self.transforms(image).unsqueeze(0)
        seg = torch.from_numpy(seg).unsqueeze(0)

        return image, seg

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

# train_data = kidney_Loader('./data/val')
# a = DataLoader(train_data, batch_size=1, shuffle=False)
# for i,(image, target) in enumerate(a):
#     print(image.shape, target.shape)
#     print(f'{i} th sample')
#     print(image.max(), image.min())  # torch.Size([2, 1, 128, 128, 128]) torch.Size([2, 1, 128, 128, 128])
#     print(target.max(),target.min())
