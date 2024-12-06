import argparse
import os
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import random
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import VOCSegmentation, Food8Segmentation
import utils
from utils import ext_transforms as et
from utils import model_summary as ms
from metrics import StreamSegMetrics
from network import UNet

def get_argparser():
    parser = argparse.ArgumentParser()

    backbone_choices = ['resnet18', 'resnet50', 'vgg16', 'mobilenetv2']
    dataset_choices = ['voc', 'food8']
    lr_policy_choices = ['step']
    loss_fn_choices = ['cross_entropy']

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset_name", type=str, default='voc',
                        choices=dataset_choices, help="dataset name")
    parser.add_argument("--num_classes", type=int, default=21,
                        help="num classes (defaults: 21)")

    # Model Options
    parser.add_argument("--backbone_name", type=str, default='resnet18',
                        choices=backbone_choices, help='backbone model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help='apply separable conv (default: False)')
    
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=5e3,
                        help="epoch number (default: 5k)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--pretrained_lr", type=float, default=1e-5,
                        help="learning rate for backbone (default: 1e-5)")
    parser.add_argument("--lr_policy", type=str, default='step', choices=lr_policy_choices,
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument("--val_batch_size", type=int, default=32,
                        help='batch size for validation (default: 32)')
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--ckpt", default="checkpoints/best_mobilenetv2_VOC.pth", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=loss_fn_choices, help="You may add different loss types here")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--vis_num_samples", type=int, default=5,
                        help='number of samples for visualization (default: 5)')
                        
    return parser

def get_train_val_dataset(opts):
    """Returns Dataset And Augmentation.
    """
    if opts.dataset_name == 'voc':
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtRandomBox(0,(50, 100), p=0.85),
            et.ExtRandomBox(0,(50, 100), p=0.85),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtRandomBox(0,(50, 100), p=0.85),
                et.ExtRandomBox(0,(50, 100), p=0.85),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtRandomBox(0,(50, 100), p=0.85),
                et.ExtRandomBox(0,(50, 100), p=0.85),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, image_set='train', transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, image_set='val', transform=val_transform)
    elif opts.dataset_name == 'food8':
        pass
    else:
        raise ValueError(f"Unknown dataset {opts.dataset_name}")

    return train_dst, val_dst

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_train_val_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          ("VOC" if opts.dataset_name == 'voc' else "COCO", len(train_dst), len(val_dst)))
    
    model = UNet(backbone_name=opts.backbone_name,
                 use_separable_conv=opts.separable_conv)
    print("Number of parameters: %d" % ms.count_parameters(model))
    utils.set_bn_momentum(model, momentum=0.01)

    metrics = StreamSegMetrics(opts.num_classes)
    
    # Set up optimizer 
    optimizer = torch.optim.Adam([{'params': model.get_pretrained_parameters(), 'lr': opts.pretrained_lr},
                                  {'params': model.get_random_initialized_parameters()}], lr=opts.lr)

    # Set up Learning Rate Policy
    if opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.9)

    # Set up criterion
    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    utils.mkdir('checkpoints')

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        print()
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples, np.int32)

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                     (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0
            
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return

        # evaluation after each epoch
        save_ckpt('checkpoints/latest_%s_%s_occlusion.pth' %
                 (opts.backbone_name, opts.dataset_name))
    
        print("validation...")
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id)

        print(metrics.to_str(val_score))

        if val_score['Mean IoU'] > best_score:  # save best model
            best_score = val_score['Mean IoU']
            print("new best mIOU: ", best_score)
            save_ckpt('checkpoints/best_%s_%s_occlusion.pth' %
                     (opts.backbone_name, opts.dataset_name))
            
if __name__ == '__main__':
    main()
