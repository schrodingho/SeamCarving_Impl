import numpy as np
import torch
from utils.gradcam_clip import gradCAM
import clip
from PIL import Image
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.UI import CamUI

def CLIP_Extractor(input_img_path, input_text):
    clip_model = "RN50"
    saliency_layer = "layer4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)
    image_src = Image.open(input_img_path)
    image_np = np.float32(image_src) / 255
    ori_height, ori_width = np.asarray(image_src).shape[:2]
    image_input = preprocess(image_src).unsqueeze(0).to(device)
    text_input = clip.tokenize([input_text]).to(device)
    attn_map = gradCAM(
        model.visual,
        image_input,
        model.encode_text(text_input).float(),
        getattr(model.visual, saliency_layer)
    )

    attn_map = attn_map.squeeze().detach().cpu().numpy()
    attn_map = normalize(attn_map)
    attn_map = resize_back(attn_map, ori_height, ori_width)
    visualization = show_cam_on_image(image_np, attn_map, use_rgb=True)
    cur = Image.fromarray(visualization)
    ui = CamUI(attn_map, image_np, cur)
    ui.run_UI()
    img_dst_path, mask_pkl_path = ui.return_dst_path()

    return img_dst_path, mask_pkl_path


def normalize(x):
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def resize_back(grayscale_cam, ori_height, ori_width):
    return cv2.resize(grayscale_cam, (ori_width, ori_height))




