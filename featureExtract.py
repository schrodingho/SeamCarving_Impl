from utils.UI import CamUI
import torchvision
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

"""
Step 1 -> Step 3
"""

def extractFeature(input_img_path):
    """
    :param input_img_path: the path of the image
    :return:
    """
    image = np.array(Image.open(input_img_path))
    image_float_np = np.float32(image) / 255
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.unsqueeze(0)

    model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)

    cur = Image.fromarray(visualization)
    ui = CamUI(grayscale_cam, image_float_np, cur)
    ui.run_UI()
    # TODO: cache folder should be created
    # TODO: block this if not saved
    img_dst_path, mask_pkl_path = ui.return_dst_path()

    return img_dst_path, mask_pkl_path

# if __name__ == '__main__':
#     input_img_path = './images/baseline.png'
#     print(extractFeature(input_img_path))

