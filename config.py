# coding: utf-8
# @Author: Salt
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = 'data'
BATCH_SIZE = 1
NUM_WORKERS = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
SAVE_MODEL = True
CHECKPOINT = "./checkpoints/weights.pth.tar"
