import torch
import numpy as np
import os.path as osp
from .utils.saveload import save_model, load_model
from .utils.visualize import visualize_ranked_results
from reid.metrics import accuracy, compute_distance_matrix, eval_market1501, open_world_eval
from torch.utils.tensorboard import SummaryWriter


class ReIDEngine(object):

    def __init__(self, model):
        self.model = model

    @classmethod
    def load_model(cls, model_class, model_path):
        """Loads state dict in model_path to the given model_class object

        Args:
            model_class (nn.Module): The feature extractor network, i.e. ResNet50 
            model_path (str): path to the pre-trained model weights (.pth file)

        Returns:
            nn.Module: the given model class with the pre-trained weights loaded
        """
        load_model(model_class, model_path)
        print('=> Loaded model')
        return cls(model_class)


    def load_gallery(self, data_loader):
        """Loads the pre-defined gallery images to memory.

        Args:
            data_loader (DataLoader): The pytorch dataloader object with the gallery images.
        """
        self.model.eval()
        g_features, g_pids, g_camids, g_img_paths = self.__extract_features(data_loader)
        self.g_features = g_features
        self.g_pids = g_pids
        self.g_camids = g_camids
        self.g_img_paths = g_img_paths
        

    def run_query(self, probe_imgs, threshold=1, device="cuda", dist_metric="cosine"):
        """Given a batch of probe images, retrieves the predicted pIDs by querying the
        reID model.

        Args:
            probe_imgs (nn.Tensor): input set of query images.
            threshold (int): minimum similarity score to assign pID to query

        Returns:
            list: the predicted pIDs for each query image in probe_imgs
        """
        self.model.eval()
        probe_ids = []

        # Get features of probe image
        probe_imgs = probe_imgs.to(device)
        q_features = self.model(probe_imgs)
        q_features = q_features.detach().cpu()

        # Calculate distances to gallery instances
        distmat = compute_distance_matrix(q_features, self.g_features, metric=dist_metric)
        print('Done, obtained {}-by-{} matrix'.format(distmat.size(0), distmat.size(1)))
        distmat = distmat.numpy()

        # Sort rankings for each query from most similar to least (by increasing distance)
        for q_idx, probe_dists in enumerate(distmat):
            indices = np.argsort(probe_dists)
            best_match_idx = indices[0]
            best_match_similarity_dist = probe_dists[best_match_idx]

            similar_match = best_match_similarity_dist <= threshold
            
            if similar_match:
                probe_ids.append((self.g_pids[best_match_idx], indices))
            else:
                probe_ids.append((None, indices))
        return probe_ids

        
    def forward_backward(self, imgs, pids):
        return NotImplementedError()

    def run(
        self,
        train_loader=None,
        open_set_loader=None,
        query_loader=None,
        gallery_loader=None,
        dataset=None,
        device="cuda",
        optimizer=None,
        scheduler=None,
        dist_metric="euclidean",
        save_dir="saved_models",
        ranks=[1, 5, 10, 20, 50],
        test_only=False,
        visrank=False,
        visrank_topk=10,
        top_correct_ranks=1,
        start_epoch=0,
        max_epochs=50,
        print_frequency=10,
        eval_frequency=0,
        open_eval=False,
        thresholds=[0.1,0.2,0.3]
    ):
        """Function used to train and test the reID model"""
        self.train_loader = train_loader
        self.open_set_loader = open_set_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.dataset = dataset
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dist_metric = dist_metric
        self.save_dir = save_dir
        self.ranks = ranks
        self.test_only = test_only
        self.visrank = visrank
        self.visrank_topk = visrank_topk
        self.top_correct_ranks = top_correct_ranks
        self.epoch = start_epoch
        self.epochs = max_epochs
        self.print_frequency = print_frequency
        self.eval_frequency = eval_frequency
        self.open_eval = open_eval
        self.thresholds = thresholds
        self.writer = None

     
        if test_only:
            print('=> Testing only')
            self._test()
            return
      
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        print('=> Starting training')
        num_batches = len(train_loader)
        for self.epoch in range(start_epoch+1, max_epochs+1):
            self._train(
                num_batches=num_batches
            )

            # Depending on the evaluation frequency, evaluate and save model.
            if eval_frequency > 0 \
                and (self.epoch) % eval_frequency == 0 \
                and (self.epoch) != self.epochs:
                self._test()
                save_model(self.model, save_dir, self.epoch)
        if self.epochs > 0:
            print('=> Final test')
            self._test()
            save_model(self.model, save_dir, self.epoch)

        if self.writer is not None:
            self.writer.close()


    def _train(
        self,
        num_batches
    ):
        running_loss = 0.0
        running_batch_size = 0
        for i, (imgs, pids, _, _) in enumerate(self.train_loader):
            self.model.train()
        
            loss_summary, batch_loss, batch_size = self.forward_backward(imgs, pids)
            running_loss += batch_loss
            running_batch_size += batch_size
            
            if (i + 1) % self.print_frequency == 0:
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    '{losses}\t'
                    '({epoch_loss})\t'
                    'lr {lr:.6f}'.format(
                        self.epoch,
                        self.epochs,
                        i + 1,
                        num_batches,
                        losses=loss_summary,
                        epoch_loss=running_loss/running_batch_size,
                        lr=self.optimizer.param_groups[-1]['lr']
                    )
                )

            if self.writer is not None:
                n_iter = self.epoch * num_batches + (i+1)
                for name, val in loss_summary.items():
                    self.writer.add_scalar("Train/" + name, val, n_iter)
                self.writer.add_scalar("Train/lr", self.optimizer.param_groups[-1]['lr'], n_iter)

        self.scheduler.step()


    def _test(self):
        self.model.eval()

        print('Extracting features from gallery set ...')
        gf, gpids, gcamids, _ = self.__extract_features(self.gallery_loader, self.device)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
        print('Extracting features from query set ...')
        qf, qpids, qcamids, _ = self.__extract_features(self.query_loader, self.device)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))
        print(
        'Computing distance matrix with metric={} ...'.format(self.dist_metric)
        )
        distmat = compute_distance_matrix(qf, gf, self.dist_metric)
        print('Done, obtained {}-by-{} matrix'.format(distmat.size(0), distmat.size(1)))
        distmat = distmat.numpy()

        print('Computing CMC and mAP ...')
        cmc, mAP = eval_market1501(
            distmat,
            qpids,
            gpids,
            qcamids,
            gcamids
        )
        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        ranks=[1, 5, 10, 20]
        cmc_curve = {}
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            name = "rank_"+str(r)
            cmc_curve[name] = cmc[r-1]

        if self.writer is not None:
            self.writer.add_scalars("Test/CMC", cmc_curve, self.epoch)
            self.writer.add_scalar("Test/mAP", mAP, self.epoch)
            

        if self.visrank:
            visualize_ranked_results(
                distmat,
                (self.dataset.query, self.dataset.gallery),
                save_dir=osp.join(self.save_dir, 'visrank'),
                topk=self.visrank_topk,
                top_correct_ranks=self.top_correct_ranks
            )


    def __extract_features(self, data_loader, device="cuda"):
        f_ = []
        pids_ = []
        camids_ = []
        img_paths_ = []
        for i, (imgs, pids, camids, img_paths) in enumerate(data_loader):
            imgs = imgs.to(device)
            features = self.model(imgs)
            features = features.detach().cpu()
            f_.append(features)
            pids_.extend(pids.tolist())
            camids_.extend(camids.tolist())
            img_paths_.extend(img_paths)


        f_ = torch.cat(f_, 0)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        img_paths_ = np.asarray(img_paths_)
        return f_, pids_, camids_, img_paths_