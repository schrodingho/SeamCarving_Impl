# Student Workspace for CS4365 Applied Image Processing
## Environment Setup
1. Install [Conda](https://www.anaconda.com/download/) on your machine (Preferred Linux-x64).
2. Create a Conda environment with the given environment.yml: `conda env create -f environment.yml`
3. Activate the environment: `conda activate final_project`
4. Install more packages via pip (Trying to avoid the conda-forge channel):
    - `pip install imageio-ffmpeg==0.4.7`
    - `pip install grad-cam`
    - `pip uninstall opencv-python` (Please uninstall the opencv-python from grad-cam first, otherwise it will cause some unexpected problems) 
    - `pip install opencv-python-headless`
    - `pip install ftfy regex tqdm`
    - `pip install git+https://github.com/openai/CLIP.git`
5. When you run the program, the pretrained model will be downloaded automatically

## Usage
1. Easy running (See all implemented features with the example image) 
```bash
# Dogcat example (Please run in the terminal))
python main.py --image ./Example/dogcat.jpg --text "Find the cat" --mask_t 0.5 --new_width 539 --new_height 380 --animation --strategy --save_grid 30
```
- Other examples:
```bash
# Tower example (carve vertically without text)
python main.py --image ./Example/tower.png --mask_t 0.5 --new_width 500 --new_height 968
# Boat example (with text)
python main.py --image ./Example/pietro.jpg --text "Boat on the river" --mask_t 0.5 --new_width 500 --new_height 400
# Boat example (without text)
python main.py --image ./Example/pietro.jpg --mask_t 0.5 --new_width 500 --new_height 400
```

2. Customization
```bash
python main.py --image <input_image_path> [--text <input_text>] [--mask_t <mask_threshold>] --new_width <new_width> --new_height <new_height> [--animation] [--strategy] [--save_grid <fig_size>]
```
- `--image`: The relative or absolute path of the input image (required)
  - Example: `--image ./Example/dogcat.jpg`
- `--text`: The text for finding the object in the image (optional, if not provided, the feature map is from the VGG)
  - Example: `--text "Find the dog"`
- `--mask_t`: The threshold for the mask (default: 0.5, range: (0, 1)) (optional, recommend to use, if not provided, the seam carving energy map will be the generated feature map)
  - Example: `--mask_t 0.6`
- `--new_width`: The new width of the image (required, should be smaller than the original width)
  - Example: `--new_width 400`
- `--new_height`: The new height of the image (required, should be smaller than the original height)
  - Example: `--new_height 400`
- `--animation`: Visualize the steps of the carving (optional)
- `--strategy`: Use the new strategy for the orientation of the triangle diagonals (optional)
- `--save_grid`: Save the triangle grid (optional) (**Not recommended**, it can be slow for a large size image)
  - Example: `--save_grid 40` (we recommend to set as 30, 40, 50, (larger will be slower but clearer))

## File Structure
- [`main.py`](main.py): The main entry point of the program, image is loaded and processed here, final result is also saved and displayed here. (Basic feature 1, 9)
- [`feat_extractor.py`](./utils/feat_extractor.py): Run a pre-trained CNN for image detection and then extract feature map from the CNN by using Grad-CAM. (Basic feature 2, 3)
- [`UI.py`](./utils/UI.py): Modify the feature map by using tkinter. (Basic feature 4)
- [`seam_carving.py`](./utils/seam_carving.py): The implementation of seam carving. (Basic feature 5)
- [`create_grid.py`](./utils/animation.py): Create pixel grid and vectorize remaining pixels by replacing them by triangles pairs. (Basic feature 6)
  - This file also includes another strategy for the orientation of the triangle diagonals. (Extended feature 3)
  - This file also contains the drawing of the triangle grid.
- [`move_back.py`](./utils/move_back.py): Move the vectors back the original positions by uncarving the previously removed columns(rows) (Basic feature 7)
- [`interpolation.py`](./utils/interpolation.py): Interpolate the pixel values of the image by using the vectorized pixels. (Basic feature 8)
- [`animation.py`](./utils/animation.py): Visualize the steps of the carving by using matplotlib. (Extended feature 1)
- [`clip_extractor.py`](./utils/clip_extractor.py): Use the another pretrained model CLIP to extract text and image, use the text embedding as the target to calculate the gradient based on the image. (Extended feature 2)
- [`gradcam_clip.py`](./utils/gradcam_clip.py): A new Grad-CAM for CLIP input. (Extended feature 2)

## 3rd party or open-source code
1. Image loading and saving:
   - [`PIL`](https://github.com/python-pillow/Pillow)
   - [`opencv-python-headless`](https://github.com/opencv/opencv-python)
   - [`imageio`](https://github.com/imageio/imageio)
   - [`dill`](https://github.com/uqfoundation/dill)
2. Pretrained CNN model or other pretrained model:
   - [`ResNet`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
   - [`VGG`](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
   - [`CLIP`](https://github.com/openai/CLIP)
3. Grad-CAM
   - [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam)
   - [`CLIP-grad-cam`](https://github.com/kevinzakka/clip_playground)
4. Viusalization
   - [`matplotlib`](https://matplotlib.org/stable/)
   - [`tkinter`](https://docs.python.org/3/library/tkinter.html)

## Author Info
- Name: Dinghao Xue
- StudentID: 5725135