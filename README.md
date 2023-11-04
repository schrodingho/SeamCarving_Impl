# Student Workspace for CS4365 Applied Image Processing

Use this workspace to solve your assignments and projects during the course CS4365 Applied Image Processing.

**Commit often** (at least once a day if you make any changes) and provide **meaningful commit messages** to document your progress while solving the programming tasks.

## Environment Setup
1. Install [Conda](https://www.anaconda.com/download/) on your machine (Preferred Linux-x64).
2. Create a Conda environment with the given environment.yml: `conda env create -f environment.yml`
3. Activate the environment: `conda activate final_project`
4. Install more packages via pip (Trying to avoid the conda-forge channel):
    - `pip install imageio-ffmpeg==0.4.7`
    - `pip install opencv-python-headless==4.8.1.78`
    - `pip install grad-cam`
    - `pip install ftfy regex tqdm`
    - `pip install git+https://github.com/openai/CLIP.git`
5. When you run the program, the pretrained model will be downloaded automatically

## Usage
1. Easy running (all default settings)
```bash
python main.py
```
2. Customization
```bash
python main.py --image <input_image_path> --text <input_text> --mask <Use feature map as mask> --animation <Show animation> --strategy <Triangle strategy>
```
- `--image`: The relative or absolute path of the input image (required)
- `--text`: The text for finding the object in the image (optional)
- `--strategy`: The strategy for orientation of the triangle diagonals

## File Structure
- [`main.py`](main.py): The main entry point of the program, image is loaded and processed here, final result is also saved and displayed here. (Basic feature 1, 9)
- [`featureExtract.py`](./utils/featureExtract.py): Run a pre-trained CNN for image detection and then extract feature map from the CNN by using Grad-CAM. (Basic feature 2, 3)
- [`UI.py`](./utils/UI.py): Modify the feature map by using tkinter. (Basic feature 4)
- [`seamCarving.py`](./utils/seamCarving.py): The implementation of seam carving. (Basic feature 5)
- [`create_grid.py`](./utils/animation.py): Create pixel grid and vectorize remaining pixels by replacing them by triangles pairs. (Basic feature 6)
- (Basic feature 7)
- [`interpolation.py`](./utils/interpolation.py): Interpolate the pixel values of the image by using the vectorized pixels. (Basic feature 8)
- [`animation.py`](./utils/animation.py): Visualize the steps of the carving by using matplotlib. (Extended feature 1)
- [`additionalExtractor.py`](./utils/additionalExtractor.py): Use the another pretrained model CLIP to extract text and image features. (Extended feature 2)
- [`gradcam_clip.py`](./utils/gradcam_clip.py): A new Grad-CAM for CLIP input. (Extended feature 2)


## 3rd party or open-source code
1. Image loading and saving:
   - [`PIL`](https://github.com/python-pillow/Pillow)
   - [`opencv-python-headless`](https://github.com/opencv/opencv-python)
   - [`imageio`](https://github.com/imageio/imageio)
   - [`dill`](https://github.com/uqfoundation/dill)
2. Pretrained CNN model or other pretrained model:
   - [`ResNet`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
   - [`CLIP`](https://github.com/openai/CLIP)
3. Grad-CAM
   - [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam)
   - [`CLIP-grad-cam`](https://github.com/kevinzakka/clip_playground)
4. Viusalization
   - [`matplotlib`](https://matplotlib.org/stable/)
   - [`tkinter`](https://docs.python.org/3/library/tkinter.html)