import argparse
import os.path as osp

import torch

from reid.datasets import get_dataset, get_dataloaders, get_dataset_names
from reid.models import get_model, get_model_names
from reid.engine import TripletEngine




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss on ImageNet")

    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=get_dataset_names())
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--verbose', action='store_true',
                        help="see output.")
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('-btrain', '--train-batch-size', type=int, default=64)
    parser.add_argument('-btest', '--test-batch-size', type=int, default=64)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-m', '--model', type=str, default='resnet50',
                        choices=get_model_names())
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--print-frequency', type=int, default=10)
    parser.add_argument('--eval-frequency', type=int, default=10)
    parser.add_argument('--dist-metric', type=str, default="euclidean")


    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(args, open_set=False)
    train_loader, gallery_loader, query_loader = get_dataloaders(args, dataset)

    num_classes = dataset.num_train_pids

    model = get_model(
        args.model,
        num_classes, 
        triplet_loss=True,
        last_conv_stride=1, 
        fc_dims=[args.features], 
        # dropout_p=args.dropout
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99)
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40,70], gamma=args.gamma
    )

    engine = TripletEngine(model)

    engine.run(
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        dataset=dataset,
        epochs=args.epochs,
        dist_metric="euclidean",
        print_frequency=args.print_frequency,
        eval_frequency=args.eval_frequency,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir="market1501imagenettripletaug",
        test_only=False
    )
