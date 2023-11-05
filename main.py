import argparse
from utils.feat_extractor import *
from utils.clip_extractor import CLIP_Extractor
from utils.seam_carving import SeamCarving
from utils.create_grid import *
from utils.interpolation import *
from utils.animation import *
from utils.move_back import *
import dill
import os
output_path = "./output"
cache_path = "./cache"

def main(args):
    # arguments parsing
    input_text = args.text
    mask_t = args.mask_t

    # if results and cache folders not exist, create them

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    print("[Step 1]: Loading the image...")

    original_img = np.array(Image.open(args.image))

    print("[Step 2, 3, 4]: Extract the feature map from the CNN using Grad-CAM, and modify the feature map...")

    if input_text:
        print("[Extended feature 2]: Using CLIP model to extract the feature map based on the text...")
        feature_map_path, mask_path = CLIP_Extractor(args.image, input_text, output_path)
    else:
        # extract features based on a pretrained CNN model and use Grad-CAM to get the feature map
        feature_map_path, mask_path = extractFeature(args.image, output_path)

    Image.open(feature_map_path).save(f"{output_path}/step_2_3_4_mod_feat_map.png")

    if mask_t is not None:
        # use the feature map as the mask, calculate the energy map based on the original image
        feature_map = dill.load(open(mask_path, "rb"))
    else:
        # use the feature map as the energy map directly (not recommended)
        feature_map = np.array(Image.open(feature_map_path).convert('P'))
        max_val = np.max(feature_map)
        feature_map = max_val - feature_map

    # save the feature map
    image_width = original_img.shape[1]
    image_height = original_img.shape[0]
    # initialize the seam carving runner

    seamCarvingRunner = SeamCarving(original_img, feature_map, mask_t=mask_t)

    # seam carving legality check
    if (args.new_width > image_width) or (args.new_height > image_height):
        raise ValueError("New image size (width and height) should not be larger than the original image size (width and height)")
    if (args.new_width < 0) or (args.new_height < 0):
        raise ValueError("New image size (width and height) should be positive")

    print("[Step 5]: Run seam carving...")

    # run seam carving with new height and width
    seamCarvingRunner.run(new_height=args.new_height, new_width=args.new_width)

    # get the carved image and show it
    carved_img_array = seamCarvingRunner.return_image()
    carved_img = Image.fromarray(carved_img_array)
    # get the original indices of each pixel in the carved image
    carved_indices = seamCarvingRunner.return_indices()

    # get the removed seam (horizontal idx and vertical idx)
    if args.animation:
        print("[Extended feature 1]: Visualize the steps of seam carving...")
        removed_width_seam, removed_height_seam = seamCarvingRunner.return_removed_seam()
        # show the animation of seam carving
        matplotlib_animation(args.image, removed_width_seam, removed_height_seam)

    # save carved_img
    carved_img.save(f"{output_path}/step_5_carved_img.png")
    # carved_img.show()

    carved_img_width = carved_img_array.shape[1]
    carved_img_height = carved_img_array.shape[0]

    print("[Step 6]: Vectorize the pixels, create vertices and triangles...")
    # select the strategy for triangle orientation, 1 for original strategy, 2 for new strategy
    if args.strategy:
        print("[Extended feature 3]: Using new strategy for orientation of the triangle diagonals")
        vertices, triangles = create_grid_strategy_2(carved_img_width, carved_img_height)
    else:
        # Default strategy
        vertices, triangles = create_grid_strategy_1(carved_img_width, carved_img_height)

    if args.save_grid is not None:
        # could be slow
        print("**Draw the triangles grid map...")
        grid_show(vertices, triangles, output_path, figure_size=args.save_grid, original=True)

    print("[Step 7]: Move the pixels back to the original positions...")
    # put the carved image pixels into original image size
    original_changed, moved_vertices = move_pixel_back(original_img, carved_img_array, carved_indices, vertices)

    Image.fromarray(original_changed).save(f"{output_path}/step_7_move_pixels_back_to_original_pos.png")

    if args.save_grid is not None:
        # could be slow
        print(" **Draw the moved triangles grid map...")
        grid_show(moved_vertices, triangles, output_path, figure_size=args.save_grid, original=False)


    print("[Step 8, 9]: Interpolate the image and save the final result...")
    # interpolate the image by using sample bilinear interpolation
    dst_img = triangle_interpolation(triangles, vertices, moved_vertices, original_changed, carved_img_array)

    # save and show the final result
    Image.fromarray(dst_img).save(f"{output_path}/step_8_9_final_result.png")
    Image.fromarray(dst_img).show()

    # delete all cache files
    for file in os.listdir(cache_path):
        os.remove(os.path.join(cache_path, file))

    print("All Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seam Carving")
    parser.add_argument('--image', type=str, required=True, help='Input image path', default='./Example/dogcat.jpg')
    parser.add_argument('--text', type=str, required=False, help='Input the text', default=None)
    parser.add_argument('--new_width', type=int, required=True, help='Seam cut number in horizontal direction', default=539)
    parser.add_argument('--new_height', type=int, required=True, help='Seam cut number in vertical direction', default=380)
    parser.add_argument('--mask_t', type=float, required=False, help='Use feature map as mask, the threshold should be 0 < t < 1', default=None)
    parser.add_argument('--animation', action='store_true', help='Show the animation of seam carving')
    parser.add_argument('--strategy', action='store_true', help='New strategy for orientation of the triangle diagonals')
    parser.add_argument('--save_grid', type=int, required=False, help='Save the triangles grid map', default=None)
    args = parser.parse_args()
    main(args)
