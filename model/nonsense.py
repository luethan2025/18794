import torch
from torchvision import models
from torchvision import transforms
import cv2
from torchvision import transforms
from torchvision.transforms.functional import crop
def create_processed_image_and_adjancent_matrix(tensor, predictions):
    
    boxes = predictions[0]['boxes']
    the_box_were_using_atm_since_Im_lazy = boxes[0]
    print(the_box_were_using_atm_since_Im_lazy[1].item())
    results = crop(tensor.squeeze(), int(the_box_were_using_atm_since_Im_lazy[1].item()), 
                   int(the_box_were_using_atm_since_Im_lazy[0].item()),
                   int(the_box_were_using_atm_since_Im_lazy[3].item() - the_box_were_using_atm_since_Im_lazy[1].item()), 
                   int(the_box_were_using_atm_since_Im_lazy[2].item() - the_box_were_using_atm_since_Im_lazy[0].item() ))
    return results


    