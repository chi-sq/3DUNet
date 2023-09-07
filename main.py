import sys,os,csv
# 导入图形组件库
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
#导入做好的界面库
from untitled import Ui_MainWindow
from MyFigure import *  # 嵌入了matplotlib的文件
from pathlib import Path
import nibabel as nib
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
class MaskWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_StyledBackground)
        self.setStyleSheet('background:rgba(0,0,0,150);')
        self.setAttribute(Qt.WA_DeleteOnClose)
        q = QHBoxLayout()
        l = QLabel("正在计算中...")
        font = QFont("Arial", 12, QFont.Bold)
        l.setFont(font)
        l.setAlignment(Qt.AlignCenter)
        l.setStyleSheet("color: white;")
        q.addWidget(l)
        q.setContentsMargins(0,0,0,0)
        self.setLayout(q)

    def show(self):
        if self.parent() is None:
            return
        parent_rect = self.parent().geometry()
        self.setGeometry(0, 0, parent_rect.width(), parent_rect.height())
        super().show()
class mythread(QThread):
    d = pyqtSignal(str)
    def __init__(self,_dir):
        super(mythread, self).__init__()
        self._dir = _dir
    def run(self):
        def crop(image):  # image refers to segmentaiotn
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
        def applicator(image_path=None):  # image_path is case_00xxx
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
            origin_depth, origin_width, origin_height = image.shape
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
            image = transforms_(image).unsqueeze(0).unsqueeze(0)  # 增加channel 和 batch
            image = image.to(config.DEVICE)
            seg_predict = model(image)  # [1, 3, 128, 128, 128]
            seg_predict = seg_predict.squeeze(0).permute(1, 2, 3, 0)  # [128,128,128,3]
            seg_predict = nn.Softmax(dim=-1)(seg_predict)
            print(seg_predict.shape)
            seg_predict = torch.argmax(seg_predict, dim=-1).to('cpu').numpy()  # [128,128,128]
            max = seg_predict.max()
            print(max)
            seg_predict = resize_volume(seg_predict, desired_depth=bottom - top, desired_height=right - left,
                                        desired_width=backward - forward)
            seg = nib.Nifti1Image(seg_predict.astype(np.int8), affine)
            image_crop = nib.Nifti1Image(image_crop, affine)
            seg_true = nib.Nifti1Image(seg_true.astype(np.int8), affine)
            print("===> saving image")
            nib.save(seg, './seg_predict.nii.gz')
            nib.save(image_crop, './image_crop.nii.gz')
            nib.save(seg_true, './seg_true_crop.nii.gz')
            print('generated seg_prediction')


        applicator(image_path=self._dir)
        self.d.emit("ok")

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        #继承(QMainWindow,Ui_MainWindow)父类的属性
        super(MainWindow,self).__init__()
        #初始化界面组件
        self.setupUi(self)
        # 创建存放nii文件路径的属性
        self.nii_path1 = ''
        self.nii_path2 = ''
        self.nii_path3 = ''

        # 定义MyFigure类的一个实例
        # self.data1 = {}
        self.F = MyFigure(width=3, height=2, dpi=100)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.F)
        self.layout.setContentsMargins(0,0,0,0)
        self.widget.setLayout(self.layout)

        # 定义MyFigure类的一个实例
        self.F1 = MyFigure(width=3, height=2, dpi=100)
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.F1)
        self.layout1.setContentsMargins(0, 0, 0, 0)
        self.widget_2.setLayout(self.layout1)
        # 定义MyFigure类的一个实例
        self.F2 = MyFigure(width=3, height=2, dpi=100)
        self.layout2 = QVBoxLayout()
        self.layout2.addWidget(self.F2)
        self.layout2.setContentsMargins(0, 0, 0, 0)
        self.widget_3.setLayout(self.layout2)
        #导入
        self.pushButton.clicked.connect(self.loadFile)
        #计算
        self.pushButton_2.clicked.connect(self.test)
        #绑定滑轨
        self.horizontalSlider.valueChanged.connect(self.bindSlider)
        # 调用遮罩
        self.mask = MaskWidget(self)
        self.mask.hide()

        self.horizontalSlider.setEnabled(False)

    def test(self):
        _dir = self.lineEdit.text()
        if _dir:
            self.p = mythread(_dir)
            self.p.d.connect(self.refrsh)
            self.p.start()
            self.mask.show()
        else:
            QMessageBox.warning(self,"警告","未选择文件夹",QMessageBox.Yes)
    def refrsh(self):
        self.horizontalSlider.setEnabled(True)
        self.mask.hide()
        self.loadNii('./image_crop.nii.gz','./seg_true_crop.nii.gz','./seg_predict.nii.gz')

    def loadFile(self):
        directory1 = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if directory1:
            self.lineEdit.setText(directory1)



    def loadNii(self,path1,path2,path3):
        self.nii_path1 = path1
        self.data_nii1 = nib.load(Path(self.nii_path1))
        self.data1 = self.data_nii1.get_fdata()



        self.nii_path2 = path2
        self.data_nii2 = nib.load(Path(self.nii_path2))
        self.data2 = self.data_nii2.get_fdata()

        self.nii_path3 = path3
        self.data_nii3 = nib.load(Path(self.nii_path3))
        self.data3 = self.data_nii3.get_fdata()

        self.showimage(slice_idx=1)
        self.horizontalSlider.setRange(1, self.data1.shape[-1])
        self.horizontalSlider.setValue(1)

    def showimage(self, slice_idx):
        self.F.axes.clear()
        self.F.axes.imshow(self.data1[:, :, slice_idx - 1], cmap='gray')
        self.F.draw()

        self.F1.axes.clear()
        self.F1.axes.imshow(self.data2[:, :, slice_idx - 1], cmap='gray')
        self.F1.draw()

        self.F2.axes.clear()
        self.F2.axes.imshow(self.data3[:, :, slice_idx - 1], cmap='gray')
        self.F2.draw()

        self.label_2.setText(f"{slice_idx}")



    def bindSlider(self):
        slice_idx = self.horizontalSlider.value()
        self.showimage(slice_idx)



if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    #创建QApplication 固定写法
    app = QApplication(sys.argv)
    # 实例化界面
    window = MainWindow()
    #显示界面
    window.show()
    #阻塞，固定写法
    sys.exit(app.exec_())