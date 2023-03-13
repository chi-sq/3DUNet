# coding: utf-8
# @Author: Salt

import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from src.resize import resize_volume
from torchvision import transforms

class kidney_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的case
        self.data_path = data_path
        self.imgs_path = os.listdir(self.data_path)
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))] )

    def augment(self, image):
        pass

    # 把CT的HU值归一化到0-1
    def normalize(self, volume):
        """Normalize the volume"""
        min = -1024.0
        max = 1413.0
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
        return volume

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        image = nib.load(os.path.join(self.data_path, image_path, 'imaging.nii.gz'))
        seg = nib.load(os.path.join(self.data_path, image_path, 'segmentation.nii.gz'))
        # get_fdata()方法转成ndarray
        image = image.get_fdata()
        seg = seg.get_fdata()
        # resize volume to 128*128*128
        image = resize_volume(image)
        seg = resize_volume(seg)
        # normalize
        image = self.normalize(image)
        seg = self.normalize(seg)
        # 转换成tensor 然后加入batch维度
        image = self.transforms(image).unsqueeze(0)
        seg = self.transforms(seg).unsqueeze(0)


        return image, seg

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


train_data = kidney_Loader('./data')
a = DataLoader(train_data, batch_size=2, shuffle=False)
for image, target in a:
    print(image.shape, target.shape)
    print(image.max(), image.min())  # torch.Size([2, 1, 128, 128, 128]) torch.Size([2, 1, 128, 128, 128])
    break
