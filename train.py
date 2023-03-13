# coding: utf-8
# @Author: Salt

'''
3d-unet train
'''

import torch
import torch.nn as nn
import torch.optim as optim
import config
from UNet_3D import UNet3D
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import kidney_Loader
from utils import save_checkpoint

torch.backends.cudnn.benchmark = True


# torch.set_default_tensor_type(torch.FloatTensor)

def train_fn(
        model, loader, opt_model, loss,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train
        # with torch.cuda.amp.autocast():
        output = model(x)
        Loss = loss(output, y)

        opt_model.zero_grad()
        Loss.backward()
        opt_model.step()


def main():
    model = UNet3D(in_channel=1, n_classes=3)
    opt_model = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    loss = nn.CrossEntropyLoss()

    train_dataset = kidney_Loader(data_path=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    # val_dataset = MapDataset(root_dir=config.VAL_DIR)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            model, train_loader, opt_model, loss
        )
        print(f"epoch{epoch + 1} done!!!")
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(model, opt_model, filename=config.CHECKPOINT)


if __name__ == "__main__":
    main()
