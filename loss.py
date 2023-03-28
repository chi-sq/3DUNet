import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# the predictions and y are tensor
def Dice_loss(predictions, y, eps=1e-8, lambda_tk=1, lambda_tu=5):
    tk_pd = torch.greater(predictions, 0)

    # compute kidney Dice
    tk_gt = torch.greater(y, 0) # compute kidney and tumor Dice
    # tk_gt = (y == 1)
    tk_dice = (2 * torch.logical_and(tk_pd, tk_gt).sum() + eps) / (
            tk_pd.sum() + tk_gt.sum() + eps
    )
    tk_dice_loss = 1 - tk_dice
    # Compute tumor Dice
    tu_pd = torch.greater(predictions, 1)
    tu_gt = torch.greater(y, 1)
    tu_dice = (2 * torch.logical_and(tu_pd, tu_gt).sum() + eps) / (
            tu_pd.sum() + tu_gt.sum() + eps
    )
    tu_dice_loss = 1 - tu_dice
    dice_loss = (lambda_tk * tk_dice_loss + lambda_tu * tu_dice_loss) / (lambda_tk + lambda_tu)
    return dice_loss
