import cProfile
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F
from network import UNet
import cv2 
import numpy as np 
from PIL import Image 
from torch.utils import data
import random 
from network.goon import Goon 
import time
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from utils import ext_transforms as et
import matplotlib.pyplot as plt 
import matplotlib 
import torchvision


def voc_cmap(N=256, normalized=False):
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

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    while not cap.isOpened(): 
        continue
    time.sleep(1)
    print(cap.isOpened())
    ret, frame = cap.read()
    print(frame.shape)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Camera", 600, 600) 
    print(ret,frame)
    start = time.time()
    sample_img = cv2.imread("snapshot.png")
    batch_tmp = transform(sample_img).unsqueeze(0) 
    torch.backends.quantized.engine = 'qnnpack'
    net = Goon(backbone_name='mobilenet', quantize=True, classes=9)
    # net = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=9)
    # net = torchvision.models.segmentation.fcn_resnet50(num_classes=9)
    # net = UNet(backbone_name="resnet18", classes=9)
    weights_path = './checkpoints/fcn_food8.pth'  # Modify this with the actual path to your weights
    net.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    net2=  torch.compile(net, mode= "max-autotune")
    net2.eval()
    model_int8=net
    
    net2.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    new_upsmaple_blocks =nn.ModuleList()
    for layer in net.upsample_blocks:
        for idx, children in enumerate(list(layer.children())):
            if idx == 1:
                torch.ao.quantization.fuse_modules(layer, [['bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
                break
        print(layer)
    #net2.upsample_blocks = new_upsmaple_blocks
    pytorch_total_params = sum(p.numel() for p in net2.parameters())
    print(pytorch_total_params)
    model_fp32_prepared = torch.ao.quantization.prepare(net2)
    model_fp32_prepared(batch_tmp)
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    #model_int8 = net2
    # pytorch_total_params = sum(p.numel() for p in model_int8.parameters())
    # # net = nn.DataParallel(net)
    # #weights = torch.load("latest_mobilenetv2_food8_occlusion.pth", map_location=torch.device('cpu'))
    # #net.load_state_dict(weights["model_state"]) 
    # #quantize_(mode, int4_weight_only())
    # #pytorch_total_params = sum(p.numel() for p in net.parameters())
    #print(pytorch_total_params)

    cmap = voc_cmap()

    for _ in range(3):   
        with torch.set_grad_enabled(True):
            batch = torch.ones(1, 3, 224, 224).normal_()
            targets = torch.ones(1, 21, 224, 224).normal_()
            start = time.time()
            out = model_int8(batch)
            #print(out)
            print(time.time() - start)
            # print(out.shape)
    model_int8(batch_tmp)
    
    print('Network initialized. Running a test batch.')
    print(f"thread count = {torch.get_num_threads()}")
    
    # out = cv2.VideoWriter('output.mp4', fourcc, 1, (224,224))
    count = 0
    total_time = 0
    # with torch.set_grad_enabled(True):
    #     frame = cv2.imread("cheesecake_original.png")
    #     batch = transform(frame).unsqueeze(0) 
    #     res = model_int8(batch)['out']
    #     res_detached = res[0].detach().numpy()
    #     class_indices = np.argmax(res_detached, axis=0)
    #     class_values = np.max(res_detached, axis=0)
    #     print(cmap[class_indices])
    #     res = cv2.addWeighted(cmap[class_indices], 0.7, frame, 0.3, 0)
    #     cv2.imwrite("fcn_cheesecake_overlay.png", res)

    #     frame = cv2.imread("donut_original.png")
    #     batch = transform(frame).unsqueeze(0) 
    #     res = model_int8(batch)['out']
    #     res_detached = res[0].detach().numpy()
    #     class_indices = np.argmax(res_detached, axis=0)
    #     class_values = np.max(res_detached, axis=0)
    #     res = cv2.addWeighted(cmap[class_indices], 0.7, frame, 0.3, 0)
    #     cv2.imwrite("fcn_donut_overlay.png", res)

    #     frame = cv2.imread("salad_original.png")
    #     batch = transform(frame).unsqueeze(0) 
    #     res = model_int8(batch)['out']
    #     res_detached = res[0].detach().numpy()
    #     class_indices = np.argmax(res_detached, axis=0)
    #     class_values = np.max(res_detached, axis=0)
    #     res = cv2.addWeighted(cmap[class_indices], 0.7, frame, 0.3, 0)
    #     cv2.imwrite("fcn_salad_overlay.png", res)

    #     frame = cv2.imread("nugget_original.png")
    #     batch = transform(frame).unsqueeze(0) 
    #     res = model_int8(batch)['out']
    #     res_detached = res[0].detach().numpy()
    #     class_indices = np.argmax(res_detached, axis=0)
    #     class_values = np.max(res_detached, axis=0)
    #     res = cv2.addWeighted(cmap[class_indices], 0.7, frame, 0.3, 0)
    #     cv2.imwrite("fcn_nugget_overlay.png", res)

    while (cap.isOpened()):
        with torch.set_grad_enabled(True):
            
            start1 = time.time()
            ret, frame = cap.read()
            frame = cv2.resize(frame[:, 80:560], (224, 224))
            # frame = cv2.resize(frame[:, 420:1500], (224, 224))
            # frame = cv2.flip(frame, 0) 

            batch = transform(frame).unsqueeze(0) 
            res = model_int8(batch)

            res_detached = res[0].detach().numpy()
            # print(res_detached.shape)
            # print(res_detached[:,0,0])
            class_indices = np.argmax(res_detached, axis=0)
            class_values = np.max(res_detached, axis=0)

            res = cv2.addWeighted(cmap[class_indices], 0.7, frame, 0.3, 0)
            cv2.imshow('Camera', cv2.resize(res, (600, 600), 
               interpolation = cv2.INTER_LINEAR))
            end1 = time.time()
            print(f"frame time = {end1 - start1}")
            count += 1
            total_time += end1 - start1
            # # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                print(f"avg time = {total_time/count}")
                # cv2.imwrite("cheesecake_original.png", frame)
                # cv2.imwrite("cheesecake_overlay.png", res)
                break

    #print(f"elapsed time = {end - start}")
    print('fasza.')

cap.release()
# out.release()
cv2.destroyAllWindows()
