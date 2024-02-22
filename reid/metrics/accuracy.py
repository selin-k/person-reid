
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Imported from:
        - https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/accuracy.py

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    # get the max k within all the k ranks to find a view for.
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    # get maxk largest elements along dimension 1 sorted in decreasing order.
    # pred are the indices of these..
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        # select the first k predictions and sum up number of correct predictions.
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # num of correct predictions/batch size * 100 is the accuracy percentage.
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res
