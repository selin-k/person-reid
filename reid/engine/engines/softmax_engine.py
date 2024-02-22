from reid.engine import ReIDEngine
from reid.metrics import accuracy
from reid.losses import CrossEntropyLoss

class SoftmaxEngine(ReIDEngine):
    """Used to train the classification model for our re-ID engine.
    Passes the given inputs through the feature extraction network,
    computes the cross entropy loss and updates network weights through
    backpropagation.

    Imported from: "https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/image/softmax.py"
    
    """

    def __init__(
        self,
        model,
        label_smoothing=0.1
    ):
        self.model = model
        self.criterion = CrossEntropyLoss(model.num_classes, epsilon=label_smoothing)
        
        super(SoftmaxEngine, self).__init__(model)
        

    def forward_backward(self, imgs, pids):
        imgs = imgs.to(self.device)
        pids = pids.to(self.device)

        outputs = self.model(imgs)
        loss = self.criterion(outputs, pids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': accuracy(outputs, pids)[0].item()
        }
        batch_loss = loss.item() * imgs.size(0)
        batch_size = imgs.size(0)

        return loss_summary, batch_loss, batch_size