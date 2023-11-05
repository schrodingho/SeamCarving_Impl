from utils.UI import CamUI
import torchvision
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import vgg13_bn
"""
Step 2: Run pre-trained CNN VGG13_bn
Step 3: Extract feature map using Grad-CAM (refer to: https://github.com/jacobgil/pytorch-grad-cam)
Step 4: Run UI to modify the feature map
"""

"""
3rd-party code:
PIL: https://github.com/python-pillow/Pillow
cv2: https://github.com/opencv/opencv-python
VGG: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam
"""
def extractFeature(input_img_path, output_path):
    """
    :param input_img_path: the path of the image
    :return: feat_img_path, mask_pkl_path
    """
    image = np.array(Image.open(input_img_path))
    image_float_np = np.float32(image) / 255
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)

    # Load Pretrained CNN model VGG13_bn
    model = vgg13_bn(pretrained=True)
    target_layers = [model.features[-1]]

    # Run gradcam
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)

    cur = Image.fromarray(visualization)
    # save the feature map
    feat_img_path = output_path + "/step_3_feat_map.png"
    cur.save(feat_img_path)

    # Run UI to modify feature map
    ui = CamUI(grayscale_cam, image_float_np, cur)
    ui.run_UI()

    feat_img_path, mask_pkl_path = ui.return_dst_path()

    return feat_img_path, mask_pkl_path

