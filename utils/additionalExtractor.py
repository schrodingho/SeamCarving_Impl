import numpy as np
import torch
from utils.gradcam_clip import gradCAM
import clip
from PIL import Image
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.UI import CamUI

def CLIP_Extractor(input_img_path, input_text):
    # Use ResNet50 as the base model of CLIP
    clip_model = "RN50"
    saliency_layer = "layer4"
    device = "cpu"

    # load CLIP model
    model, preprocess = clip.load(clip_model, device=device, jit=False)
    image_src = Image.open(input_img_path)
    image_np_float = np.float32(image_src) / 255
    original_height, original_width = np.asarray(image_src).shape[:2]

    # Preprocess the image (Image will be resized by the processor for feeding into the model)
    image_input = preprocess(image_src).unsqueeze(0).to(device)

    # Tokenize the text
    text_input = clip.tokenize([input_text]).to(device)

    # Run Grad-CAM
    # In CLIP, the image and text are embedded into the same space, so we can use the text embedding as the target to calculate the gradient
    feature_map = gradCAM(
        model.visual,
        image_input, # use the image as the input
        model.encode_text(text_input).float(), # use the text embedding as the target
        getattr(model.visual, saliency_layer)
    )

    feature_map = feature_map.squeeze().detach().cpu().numpy()

    # Normalize the feature map to 0 -> 1
    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))

    # feature_map = normalize(feature_map)
    feature_map = resize_back(feature_map, original_height, original_width)
    visualization = show_cam_on_image(image_np_float, feature_map, use_rgb=True)
    cur = Image.fromarray(visualization)

    # Run UI to modify feature map
    ui = CamUI(feature_map, image_np_float, cur)
    ui.run_UI()
    img_dst_path, mask_pkl_path = ui.return_dst_path()

    return img_dst_path, mask_pkl_path


def normalize(x):
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def resize_back(grayscale_cam, original_height, original_width):
    # resize the feature map back to the original size
    return cv2.resize(grayscale_cam, (original_width, original_height))




