import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy

def matplotlib_animation(img_path, recorded_width_seam, recorded_height_seam):
    """
    :param img_path: path of the original image
    :param recorded_width_seam: seams moved in vertical direction
    :param recorded_height_seam: seams moved in horizontal direction
    """
    img_array = np.array(Image.open(img_path))
    original_width = img_array.shape[1]
    original_height = img_array.shape[0]
    width = original_width
    height = original_height
    plt.ion()
    figure, ax = plt.subplots()
    im = ax.imshow(img_array.astype('uint8'))
    white_column = None
    for idx in range(len(recorded_width_seam)):
        width -= 1
        # Fill the removed column with white column
        white_column = np.full((height, original_width - width, 3), (255, 255, 255), dtype=np.uint8)
        if original_width - width == 1:
            white_column_2 = None
        else:
            white_column_2 = np.full((height, original_width - width - 1, 3), (255, 255, 255), dtype=np.uint8)
        new_img = np.empty((height, width, 3))
        for y, x in recorded_width_seam[idx]:
            img_array[y, x] = (0, 0, 0)
            new_img[y] = np.delete(img_array[y], x, axis=0)

        # Show the seam
        if white_column_2 is not None:
            temp_array = np.append(copy.deepcopy(img_array), white_column_2, axis=1)
        else:
            temp_array = copy.deepcopy(img_array)
        im.set_data(temp_array.astype('uint8'))
        figure.canvas.draw()
        figure.canvas.flush_events()

        img_array = new_img.copy()
        temp_arr = np.append(copy.deepcopy(new_img), white_column, axis=1)
        im.set_data(temp_arr.astype('uint8'))
        figure.canvas.draw()
        figure.canvas.flush_events()

    img_array = np.swapaxes(img_array, 0, 1)
    height, width = width, height
    original_height, original_width = original_width, original_height

    for idx in range(len(recorded_height_seam)):
        width -= 1
        # Fill the removed row with white row
        white_row = np.full((original_width - width, height, 3), (255, 255, 255), dtype=np.uint8)
        if original_width - width == 1:
            white_row_2 = None
        else:
            white_row_2 = np.full((original_width - width - 1, height, 3), (255, 255, 255), dtype=np.uint8)
        new_img = np.empty((height, width, 3))
        for y, x in recorded_height_seam[idx]:
            img_array[y, x] = (0, 0, 0)
            new_img[y] = np.delete(img_array[y], x, axis=0)

        temp_img = np.swapaxes(img_array, 0, 1)
        if white_row_2 is not None:
            temp_img = np.append(temp_img, white_row_2, axis=0)
        if white_column is not None:
            temp_img = np.append(temp_img, white_column, axis=1)
        im.set_data(temp_img.astype('uint8'))
        figure.canvas.draw()
        figure.canvas.flush_events()
        img_array = new_img.copy()
        temp_img = np.swapaxes(img_array, 0, 1)
        temp_img = np.append(temp_img, white_row, axis=0)
        if white_column is not None:
            temp_img = np.append(temp_img, white_column, axis=1)
        im.set_data(temp_img.astype('uint8'))
        figure.canvas.draw()
        figure.canvas.flush_events()

    plt.close()
    plt.ioff()

