from utils.UI import CamUI
import torchvision
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.models import vgg13_bn
image_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
dog_cat = "./images/baseline.png"
# image = np.array(Image.open(requests.get(image_url, stream=True).raw))
image = np.array(Image.open(dog_cat))
image_float_np = np.float32(image) / 255
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

input_tensor = transform(image)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.unsqueeze(0)
####
# model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]

model = vgg13_bn(pretrained=True)
target_layers = [model.features[-1]]

cam = GradCAM(model=model, target_layers=target_layers)
targets = None
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

visualization = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
cur = Image.fromarray(visualization)

Image.fromarray(visualization).show()
# ui = CamUI(grayscale_cam, image_float_np, cur)
# ui.run_UI()

