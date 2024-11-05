import torch
from torchvision import models
from torchvision import transforms
import cv2
from torchvision import transforms
from torchvision.transforms.functional import crop
def create_processed_image_and_adjancent_matrix():
    model = models.detection.fcos_resnet50_fpn(weights= models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    img = cv2.imread("HorseMan.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    transform = transforms.ToTensor()
    tensor = transform(img_rgb).unsqueeze(0)
    print(type(tensor))
    predictions = model(tensor)
    #print(predictions)
    with torch.no_grad():
        predictions = model(tensor)
    boxes = predictions[0]['boxes']
    the_box_were_using_atm_since_Im_lazy = boxes[0]
    print(the_box_were_using_atm_since_Im_lazy[1].item())
    results = crop(tensor.squeeze(), int(the_box_were_using_atm_since_Im_lazy[1].item()), 
                   int(the_box_were_using_atm_since_Im_lazy[0].item()),
                   int(the_box_were_using_atm_since_Im_lazy[3].item() - the_box_were_using_atm_since_Im_lazy[1].item()), 
                   int(the_box_were_using_atm_since_Im_lazy[2].item() - the_box_were_using_atm_since_Im_lazy[0].item() ))
    cropped_img = results.permute(1, 2, 0).numpy() * 255  # Convert to HWC format and scale
    cropped_img = cropped_img.astype('uint8')
    cropped = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite( "goat.jpg", cropped)
    
    
    
create_processed_image_and_adjancent_matrix()