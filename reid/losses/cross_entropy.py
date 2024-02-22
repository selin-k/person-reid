import torch
import torch.nn as nn

class LSRCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer (LSR).

    Label smoothing is a regularization technique for classification problems
    to prevent the model from predicting the labels too confidently during 
    training and generalizing poorly.

    References:
        - Szegedy et al. Rethinking the Inception Architecture for Computer Vision.
          CVPR 2016.
        - https://leimao.github.io/blog/Label-Smoothing/
        
    Equation: 
        y = (1 - epsilon) * targets + epsilon / num_classes.

    Args:
        - num_classes (int): number of classes.
        - epsilon (float): weight factor (value between 0 and 1).
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(LSRCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, pred, target):
        """
        Args:
            - pred (torch.Tensor): prediction matrix with shape (batch_size, num_classes)
            - targets (torch.Tensor): ground truth matrix with shape (num_classes)
        """
        # Reference: https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability
        preds = self.logsoftmax(pred)
        
        # Compute one-hot encoding of the ground truth distribution 
        targets = torch.zeros_like(preds)
        targets = targets.scatter_(1, target.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()

        # Compute LSR Cross Entropy Loss
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * preds).mean(0).sum()
        return loss