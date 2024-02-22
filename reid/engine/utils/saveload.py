import os.path as osp
import torch

from reid.engine.utils import mkdir_if_missing

def save_model(model, save_dir, epoch):
    mkdir_if_missing(save_dir)
    fpath = osp.join(save_dir, 'model.pth.tar-' + str(epoch))
    torch.save(model.state_dict(), fpath)
    print('Checkpoint saved to "{}"'.format(fpath))


def load_model(model_class, model_path):
    model_class.load_state_dict(torch.load(model_path))