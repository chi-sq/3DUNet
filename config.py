# coding: utf-8
# @Author: Salt
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = './data/train'
VAL_DIR = './data/val'
TEST_DIR = './data/test'
BATCH_SIZE = 1
NUM_WORKERS = 4
NUM_EPOCHS = 50
# optional ---> ['Cross Entropy loss','Focal loss','Dice loss','Mixed CE loss and Dice loss']
LOSS = 'Mixed CE loss and Dice loss'
LEARNING_RATE = 0.001  # paper not given default 0.0001
gamma = 2
SAVE_MODEL = True
LOAD_MODEL = True
CHECKPOINT = "./checkpoints/weights_croped_mixed_C_D_loss_new_BN.pth.tar"  # 其它都是没有BN
CHECKPOINT_SAVE = "./checkpoints/weights_croped_mixed_C_D_loss_new_BN.pth.tar"
csv_path = 'index/evaluation_mixed_C_D_loss_new_BN.csv'
