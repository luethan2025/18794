import argparse
from data import SegmentationDataset
from utils import ExtToTensor
from utils import ExtCompose
from utils import dice_loss
from network import UNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def main(opts):
    transforms = ExtCompose([
        ExtToTensor()
    ])

    dataset = SegmentationDataset(opts.data, transforms=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=opts.batch_size)
    training_loop(dataloader, opts)

def training_loop(dataloader, opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(
        in_channels=opts.in_channels,
        n_classes=opts.n_classes,
        conv_dim=opts.conv_dim
    )
    model = model.to(device)

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=opts.lr,
        weight_decay=opts.weight_decay,
        momentum=opts.momentum,
        foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, opts.mode, patience=opts.patience)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    for epoch in range(opts.num_epochs):
        model.train()
        for (images, labels) in dataloader:
            optimizer.zero_grad()
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            print(labels.unique())
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels) + dice_loss(
            #     F.softmax(outputs, dim=1).float(),
            #     F.one_hot(labels, opts.n_classes).permute(0, 3, 1, 2).float()
            # )
            optimizer.step()
            break
        break
        # scheduler.step()
        # print(f"Epoch [{epoch+1}/{opts.num_epochs}], Loss: {loss.item()}")


def create_parser():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--conv_dim', type=int, default=64)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--momentum', type=float, default=0.)

    parser.add_argument('--mode', type=str, default='max')
    parser.add_argument('--patience', type=int, default=5)

    # Data sources
    parser.add_argument('--data', type=str, default='data/dataset/test/')

    return parser

if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    main(opts)
