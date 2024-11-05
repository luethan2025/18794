import cv2
import torch
from torchvision import models
from torchvision.transforms.functional import crop

from network.mask_head import MaskHead
from network.mask_head import Conv3x3
from network.mask_head import GCN

def create_processed_image_and_adjancent_matrix(tensor, predictions):
    
    boxes = predictions[0]['boxes']
    the_box_were_using_atm_since_Im_lazy = boxes[0]
    results = crop(tensor.squeeze(), int(the_box_were_using_atm_since_Im_lazy[1].item()), 
                   int(the_box_were_using_atm_since_Im_lazy[0].item()),
                   int(the_box_were_using_atm_since_Im_lazy[3].item() - the_box_were_using_atm_since_Im_lazy[1].item()), 
                   int(the_box_were_using_atm_since_Im_lazy[2].item() - the_box_were_using_atm_since_Im_lazy[0].item() ))
    return results

if __name__ == "__main__":
    test_image = "test_image.jpg"
    img =  torch.from_numpy(
        cv2.imread(test_image)
    )
    img = img.float() / 255.0
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    # label = "apple"

    backbone = models.detection.fcos_resnet50_fpn(weights=models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)
    backbone.eval()
    Z = backbone(img)
    results = create_processed_image_and_adjancent_matrix(img, Z)
    cropped_img = results.permute(1, 2, 0).numpy() * 255  # Convert to HWC format and scale
    cropped_img = cropped_img.astype('uint8')
    cv2.imwrite("test_image_cropped.jpg", cropped_img)

    conv  = Conv3x3(3, 1)
    Z = conv(results)
    print(Z.shape)
    gcn = GCN(1, 1)
    Z = gcn(Z.detach())
