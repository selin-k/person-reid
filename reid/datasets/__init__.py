from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .cuhk03 import CUHK03


__dataset_factory = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'cuhk03': CUHK03,
}


def get_dataset_names():
    return print(list(__dataset_factory.keys()))


def init_dataset(name, *args, **kwargs):
    if name not in __dataset_factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __dataset_factory[name](*args, **kwargs)


from .dataset import ImageDataset
from .utils import build_transforms
from torch.utils.data import DataLoader

def get_dataset(dataset_name, data_dir, open_set=True, open_set_gallery_ratio=1):
    dataset = init_dataset(
        dataset_name, 
        root=data_dir
    )
    return dataset

def get_dataloaders(args, dataset):

    transform_train, transform_test = build_transforms(
        height=args.height,
        width=args.width,
        transforms=["random_flip", "random_crop", 'random_erase', 'color_jitter'],
    )

    train_loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )

    query_loader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )


    return train_loader, gallery_loader, query_loader



def get_open_set_dataloaders(args, dataset):
    _, transform_test = build_transforms(
        height=args.height,
        width=args.width
    )

    gallery_loader = DataLoader(
        ImageDataset(dataset.open_set_gallery, transform=transform_test),
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )

    query_loader = DataLoader(
        ImageDataset(dataset.open_set_probes, transform=transform_test),
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )


    return gallery_loader, query_loader