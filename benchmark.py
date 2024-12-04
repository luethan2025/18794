import argparse
import time
import torch
import numpy as np
from network import UNet

def get_argparser():
    parser = argparse.ArgumentParser()

    # Benchmark Options
    parser.add_argument("--runs", type=int, default='10')
    
    # Backbone Options
    backbone_choices = ['resnet18', 'resnet50', 'vgg16', 'mobilenetv2']
    parser.add_argument("--backbone_name", type=str, default='resnet18',
                        choices=backbone_choices, help='backbone model name')
    
    # Optimization Options
    parser.add_argument("--compile", action='store_true', default=False,
                        help='precompile model')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help='apply separable conv')
       
    return parser

def run_inference(model, opts):
    """Runs a constant number of forward pass through the model.
    Args:
        model (nn.Module): Model to benchmark.
        opts: Command line arguments.
    Returns:
        np.ndarray: Time for each run (in seconds).
    """
    if opts.compile:
        print("JIT-compiling torch model...")
        model = torch.compile(model, mode='reduce-overhead')
    model.eval()
    times = []
    with torch.no_grad():
        for run_num in range(opts.runs):
            batch =  torch.empty(1, 3, 224, 224).normal_()
            start_time = time.time()
            _ = model(batch)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"{run_num}/{opts.runs}:\t{end_time - start_time}")
    times = np.array(times)

    return times

if __name__ == "__main__":
    opts = get_argparser().parse_args()
    model = UNet(backbone_name=opts.backbone_name,
                 use_separable_conv=opts.separable_conv)
    times = run_inference(model, opts)
    print(f"Average time over {opts.runs} runs: {times.mean()}")
