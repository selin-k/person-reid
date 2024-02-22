from reid.engine import ReIDEngine
from reid.metrics import accuracy
from reid.losses import TripletLoss, CrossEntropyLoss

class TripletEngine(ReIDEngine):
    """Used to train the triplet model for our re-ID engine.
    Passes the given inputs through the feature extraction network,
    computes the cross entropy loss and triplet loss, then updates
    network weights through backpropagation.

    Imported from: "https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/image/triplet.py"
    
    """

    def __init__(
        self,
        model,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        label_smoothing=0.1
    ):
        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0

        self.model = model
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(model.num_classes, epsilon=label_smoothing)
        
        super(TripletEngine, self).__init__(model)

    def forward_backward(self, imgs, pids):

        imgs = imgs.to(self.device)
        pids = pids.to(self.device)

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.criterion_t(features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_triplet'] = loss_t.item()

        if self.weight_x > 0:
            loss_x = self.criterion_x(outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_softmax'] = loss_x.item()
            loss_summary['acc'] = accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_loss = loss.item() * imgs.size(0)
        batch_size = imgs.size(0)

        return loss_summary, batch_loss, batch_size