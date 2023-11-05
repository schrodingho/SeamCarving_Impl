import numpy as np
"""
Step 7: Move carved vectors back to original image
"""
def move_pixel_back(original_img_array, carved_img_array, carved_indices_array, vertices):
    '''
    move carved pixel(vectors) back to original image and show it
    :param original_img_array: original image array
    :param carved_img_array:  carved image array
    :param carved_indices_array:  indices of each pixel in the carved image
    :param vertices: original vertices of carved image
    :return: original image with carved pixels
    '''
    original_changed = np.zeros((original_img_array.shape[0], original_img_array.shape[1], 3))
    carved_width = carved_img_array.shape[1]
    carved_height = carved_img_array.shape[0]
    for i in range(carved_height):
        for j in range(carved_width):
            original_changed[carved_indices_array[i, j]] = carved_img_array[i, j]
    original_changed = original_changed.astype(np.uint8)

    # calculate the displacement of each pixel
    delta_indices = indices_changes(carved_indices_array)
    # move the vertices based on the displacement
    moved_vertices = vertices + delta_indices.astype(np.float32)
    return original_changed, moved_vertices

def indices_changes(indices):
    height = indices.shape[0]
    width = indices.shape[1]

    original_indices = np.meshgrid(range(width), range(height))
    row_o_indices, col_o_indices = original_indices

    # Flatten the row and column indices
    flat_row_indices = row_o_indices.ravel()
    flat_col_indices = col_o_indices.ravel()
    concatenated_indices = np.column_stack((flat_col_indices, flat_row_indices))

    indices = indices.flatten()
    indices = [list(t) for t in indices]

    # calculate the displacement of each pixel
    indices_changes = indices - concatenated_indices
    return indices_changes