import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss function
    
    References:
        - Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
        - https://gombru.github.io/2019/04/03/ranking_loss/
        - https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
        - https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
        mining (string, optional): mining technique to be used. Default is "online_hard".
    """

    def __init__(self, margin=0.3, mining="online_hard"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mining = mining


    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        """

        if self.mining == "online_hard":
            dist_an, dist_ap = self.online_hard_example_mining(inputs, targets)
        else:
            raise NotImplementedError("Unknown parameter, '{}', passed as mining technique.".format(self.mining))
        
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
    

    def euclidean_dist(self, x, y):
        """
        Args:
            x: pytorch Variable, with shape [m, d]
            y: pytorch Variable, with shape [n, d]
        Returns:
            dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    

    def online_hard_example_mining(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        Returns:
            dist_an (torch.Tensor): feature map of the hard negative example
            dist_ap (torch.Tensor): feature map of the hard positive example
        """

        n = inputs.size(0)
        dist_mat = self.euclidean_dist(inputs, inputs)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist_mat[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_mat[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        return dist_an, dist_ap
