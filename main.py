import argparse
from featureExtract import *
from additionalExtractor import CLIP_Extractor
from seamCarving import SeamCarve
from utils.create_grid import *
from utils.interpolation import *
from utils.animation import *

def main(args):
    input_text = args.text
    original_img = np.array(Image.open(args.image))
    if input_text:
        feature_map_path, mask_path = CLIP_Extractor(args.image, input_text)
    else:
        feature_map_path, mask_path = extractFeature(args.image)
    feature_map = np.array(Image.open(feature_map_path).convert('P'))
    max_val = np.max(feature_map)
    # min_val = np.min(feature_map)
    feature_map = max_val - feature_map
    image_width = original_img.shape[1]
    image_height = original_img.shape[0]
    # TODO: energy map multi method
    seamCarvingRunner = SeamCarve(original_img, feature_map)
    if (args.seam_h_num >= image_width) or (args.seam_v_num >= image_height):
        raise ValueError("Seam number should be smaller than the width or the height of the image")
    if (args.seam_h_num < 0) or (args.seam_v_num < 0):
        raise ValueError("Seam number should be positive")

    seamCarvingRunner.resize(new_height=image_height - args.seam_v_num, new_width=image_width - args.seam_h_num)

    carved_img_array = seamCarvingRunner.image()
    carved_img = Image.fromarray(carved_img_array)
    # TODO: multiprocessing animation and interpolation
    carved_indices = seamCarvingRunner.return_indices()
    removed_width_seam, removed_height_seam = seamCarvingRunner.return_removed_seam()
    matplotlib_animation(args.image, removed_width_seam, removed_height_seam)
    carved_img.show()

    def move_pixel_back(original_img_array, carved_img_array, carved_indices_array):
        original_changed = np.zeros((original_img_array.shape[0], original_img_array.shape[1], 3))
        carved_width = carved_img_array.shape[1]
        carved_height = carved_img_array.shape[0]
        for i in range(carved_height):
            for j in range(carved_width):
                original_changed[carved_indices_array[i, j]] = carved_img_array[i, j]
        original_changed = original_changed.astype(np.uint8)
        Image.fromarray(original_changed).show()
        return original_changed

    original_changed = move_pixel_back(original_img, carved_img_array, carved_indices)
    carved_img_width = carved_img_array.shape[1]
    carved_img_height = carved_img_array.shape[0]
    delta_indices = indices_changes(carved_indices)
    vertices, triangles = create_pixel_center_grid(carved_img_width, carved_img_height)
    moved_vertices = vertices + delta_indices.astype(np.float32)
    dst_img = triangle_interpolation2(triangles, vertices, moved_vertices, original_changed, carved_img_array)
    Image.fromarray(dst_img).show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seam Carving")
    parser.add_argument('--image', type=str, required=False, help='Input image path', default='./Example/dogcat.jpg')
    parser.add_argument('--text', type=str, required=False, help='Input the text', default=None)
    parser.add_argument('--seam_h_num', type=str, required=False, help='seam cut number in horizontal direction', default=50)
    parser.add_argument('--seam_v_num', type=str, required=False, help='seam cut number in vertical direction', default=30)
    parser.add_argument('--strategy', type=str, required=False, help='interpolation strategy', default=False)
    args = parser.parse_args()
    main(args)
