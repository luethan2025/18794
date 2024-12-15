from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import pickle
# from backbones_unet.model.unet import Unet
from network import UNet
from torch.utils import data
from datasets import VOCSegmentation, Food8Segmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from network import Goon
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
import torchvision 

class ExtRandomBox(object):
    def voc_cmap(N=256, normalized=False):
        """ 
        Python implementation of the color map function for the PASCAL VOC data set. 
        Official Matlab version can be found in the PASCAL VOC devkit 
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
        """
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap
        return cmap
    
    # color map for each category
    cmap = voc_cmap()

    """
    Args:
        cls (int): What class the box will belong to.
        size (tuple): Minimum and maximum size of the bounding box.
        p (float): probability of the image being flipped. Default value is 0.5.
    """
    def __init__(self, cls, size, p=0.5):
        self.cls = cls
        self.min_size, self.max_size = size
        self.p = p

        assert self.min_size > 0 and self.max_size > 0
        assert self.min_size < self.max_size
    
    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image.
            lbl (PIL Image): Label.
        Returns:
            PIL Image: Image with a box place somewhere.
            PIL Image: Image with the same box place somewhere.
        """
        if random.random() < self.p:
            w, h = img.size
            x1, y1 = random.randint(0, w - self.max_size), random.randint(0, h - self.max_size)
            x2, y2 = x1 + random.randint(self.min_size, self.max_size), y1 + random.randint(self.min_size, self.max_size)
            im = ImageDraw.Draw(img)

            r, g, b = tuple(self.cmap[self.cls])
            im.rectangle([x1, y1, x2, y2], fill=(r, g , b))

            color = int(0.2989*r + 0.5870*g + 0.1140*b) # luminance
            lbl.paste(Image.new("L", (x2 - x1, y2 - y1), color=color), (x1, y1))

        return img, lbl
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += "min_size={0}".format(self.min_size)
        format_string += ", max_size={0}".format(self.max_size)
        format_string += ", p={0}".format(self.p)
        format_string += ')'
        return format_string
class ExtAddBlackSquare:
    def __init__(self, square_size_range=(10, 50)):
        """
        Initializes the transformation.
        :param square_size_range: Tuple specifying the min and max size of the black square.
        """
        self.square_size_range = square_size_range

    def __call__(self, image,target = None):
        """
        Applies the transformation.
        :param image: Input PIL image.
        :return: Image with a black square added.
        """
        import random
        from PIL import ImageDraw

        # Get the image dimensions
        width, height = image.size
        
        # Determine the square size randomly within the specified range
        square_size = random.randint(self.square_size_range[0], self.square_size_range[1])
        
        # Randomly choose the top-left corner of the square
        top_left_x = random.randint(0, max(0, width - square_size))
        top_left_y = random.randint(0, max(0, height - square_size))

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Draw the black square
        draw.rectangle(
            [top_left_x, top_left_y, top_left_x + square_size, top_left_y + square_size],
            fill=(0, 0, 0)
        )

        return image, target
def get_argparser():
    parser = argparse.ArgumentParser()
    dataset_choices = ['voc', 'food8']

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset_name", type=str, default='food8',
                        choices=dataset_choices, help="dataset name")
    parser.add_argument("--num_classes", type=int, default=9,
                        help="num classes (default: None)")
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='quant_back_bone',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=1e5,
                        help="epoch number (default: 5k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step', 'poly'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument("--val_batch_size", type=int, default=32,
                        help='batch size for validation (default: 32)')
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--ckpt", default="checkpoints/main_food8.pth", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)#'./checkpoints/2best_quant_back_bone_VOC_os16.pth'"checkpoints/unet_res18_VOC.pth"

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="You may add different loss types here")
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
                        help='number of samples for visualization (default: 8)')
    return parser


def get_train_val_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset_name == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            ExtRandomBox(0,(50,100)),  # Add black square
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                ExtRandomBox(0,(50,100)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                ExtRandomBox(0, (50, 100)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, image_set='train', transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, image_set='val', transform=val_transform)
    elif opts.dataset_name == 'food8':
        print("bingulongulous")
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            # ExtRandomBox(0,(50,100)),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                # ExtRandomBox(0,(50,100)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = Food8Segmentation(root=opts.data_root, image_set='train', transform=train_transform)
        val_dst = Food8Segmentation(root=opts.data_root, image_set='val', transform=val_transform)
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
        for j, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and j in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results and j < 1:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    
                    if (i == 30 or i == 0 or i == 17 or i == 8): 
                        # Image.fromarray(image).save('results/food_main_%d_image.png' % img_id)
                        # Image.fromarray(target).save('results/food_main_%d_target.png' % img_id)
                        # Image.fromarray(pred).save('results/food_main_%d_pred.png' % img_id)

                        fig = plt.figure()
                        plt.imshow(image)
                        plt.axis('off')
                        plt.imshow(pred, alpha=0.7)
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        plt.savefig('results/food_fcn_%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    # TODO: you should fill the num_classes here. Don't forget to add the background class
    # opts.num_classes = 20 + 1

    #os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
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
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset_name, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    # model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=9)
    # model = torchvision.models.segmentation.fcn_resnet50(num_classes=9)
    model = Goon(classes=9, quantize=False)
    # model = UNet(backbone_name="resnet18", classes=9)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model, momentum=0.01)

    # HW
    # ================================================= please fill the blank ============================================== #
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer 
    # TODO Problem 3.1
    # please check argument parser for learning rate reference.
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr':1e-4}])

    # Set up Learning Rate Policy
    # TODO Problem 3.1
    # please check argument parser for learning rate policy.
    if opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.9)
    elif opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        

    # Set up criterion 
    # TODO Problem 3.1
    # please check argument parser for loss function.
    # in 3.3, please use CrossEntropyLoss.
    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=15, reduction='mean')
    elif opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=15, size_average=True)
    
    def save_ckpt(path):
        """ save current model
        """
        torch.save( model.state_dict(),path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        del checkpoint['final_conv.weight']
        del checkpoint['final_conv.bias']
        checkpoint['final_conv.weight'] = torch.empty(9, 16, 1, 1)
        nn.init.uniform_(checkpoint['final_conv.weight'])
        # print(checkpoint['final_conv.weight'].shape)
        checkpoint['final_conv.bias'] = torch.empty(9)
        nn.init.uniform_(checkpoint['final_conv.bias'])
        model.load_state_dict(checkpoint)
        # #model = nn.DataParallel(model)
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
        #model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples, np.int32)  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    mIoU_per_epoch = []
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # TODO Please finish the main training loop and record the mIoU
            # Problem 3.3 & Problem 3.4
            optimizer.zero_grad()
            # print(images)
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
                with open('mIoU_per_epoch.pkl', 'wb') as f:
                    pickle.dump(mIoU_per_epoch, f)
                return

        # # evaluation after each epoch
        # save_ckpt('checkpoints/latest_res_no_box_unet%s_%s_os%d.pth' %
        #             (opts.model, opts.dataset_name, opts.output_stride))
        print()
        print("main_food8")
        print()

        print("validation...")
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id)

        print(metrics.to_str(val_score))
        # TODO record mIoU with mIoU_per_epoch 
        mIoU_per_epoch.append(float(val_score['Mean IoU']))

        if val_score['Mean IoU'] > best_score:  # save best model
            best_score = val_score['Mean IoU']
            print("new best mIOU: ", best_score)
            save_ckpt('checkpoints/ignore.pth')

if __name__ == '__main__':
    main()
