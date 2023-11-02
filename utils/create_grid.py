import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
def create_warping_grid(width, height):
    # Build vertex buffer.
    num_vertices = (width + 1) * (height + 1)
    vertices = np.empty((num_vertices, 2), dtype=np.float32)

    # Fill the vertex buffer (vertices) with 2D coordinates of the pixel corners.
    for y in range(height + 1):
        for x in range(width + 1):
            vertices[y * (width + 1) + x] = np.array([x, y], dtype=np.float32)

    # Build index buffer.
    num_pixels = width * height
    num_triangles = num_pixels * 2
    triangles = np.empty((num_triangles, 3), dtype=np.int32)

    # Fill the index buffer (triangles) with indices pointing to the vertex buffer.
    for y in range(height):
        for x in range(width):
            A = y * (width + 1) + x
            B = y * (width + 1) + x + 1
            C = (y + 1) * (width + 1) + x + 1
            D = (y + 1) * (width + 1) + x
            triangle1 = np.array([A, B, C], dtype=np.int32)
            triangle2 = np.array([A, C, D], dtype=np.int32)
            triangles[(y * width + x) * 2] = triangle1
            triangles[(y * width + x) * 2 + 1] = triangle2

    # Combine the vertex and index buffers into a mesh.
    return vertices, triangles

def create_pixel_center_grid(width, height):
    # Build vertex buffer.
    num_vertices = width * height
    vertices = np.empty((num_vertices, 2), dtype=np.float32)

    # Fill the vertex buffer (vertices) with 2D coordinates of the pixel corners.
    for y in range(height):
        for x in range(width):
            vertices[y * width + x] = np.array([y + 0.5, x + 0.5], dtype=np.float32)

    # Build index buffer.
    num_pixels = (width - 1) * (height - 1)
    num_triangles = num_pixels * 2
    triangles = np.empty((num_triangles, 3), dtype=np.int32)

    # Fill the index buffer (triangles) with indices pointing to the vertex buffer.
    for y in range(height - 1):
        for x in range(width - 1):
            A = y * width + x
            B = y * width + x + 1
            C = (y + 1) * width + x + 1
            D = (y + 1) * width + x
            triangle1 = np.array([A, B, C], dtype=np.int32)
            triangle2 = np.array([A, C, D], dtype=np.int32)
            triangles[(y * (width - 1) + x) * 2] = triangle1
            triangles[(y * (width - 1) + x) * 2 + 1] = triangle2

    # Combine the vertex and index buffers into a mesh.
    return vertices, triangles

def create_pixel_center_grid(width, height, vertical=False):
    # Build vertex buffer.
    num_vertices = width * height
    vertices = np.empty((num_vertices, 2), dtype=np.float32)

    # Fill the vertex buffer (vertices) with 2D coordinates of the pixel corners.
    for y in range(height):
        for x in range(width):
            vertices[y * width + x] = np.array([y + 0.5, x + 0.5], dtype=np.float32)

    # Build index buffer.
    num_pixels = (width - 1) * (height - 1)
    num_triangles = num_pixels * 2

    if vertical:
        triangles_h = np.empty((num_triangles, 3), dtype=np.int32)
        triangles_v = np.empty((num_triangles, 3), dtype=np.int32)
        for y in range(height - 1):
            for x in range(width - 1):
                A = y * width + x
                B = y * width + x + 1
                C = (y + 1) * width + x + 1
                D = (y + 1) * width + x
                triangle1 = np.array([A, B, C], dtype=np.int32)
                triangle2 = np.array([A, C, D], dtype=np.int32)
                triangle3 = np.array([D, A, B], dtype=np.int32)
                triangle4 = np.array([D, B, C], dtype=np.int32)
                triangles_h[(y * (width - 1) + x) * 2] = triangle1
                triangles_h[(y * (width - 1) + x) * 2 + 1] = triangle2
                triangles_v[(y * (width - 1) + x) * 2] = triangle3
                triangles_v[(y * (width - 1) + x) * 2 + 1] = triangle4
        # Combine the vertex and index buffers into a mesh.
        return vertices, triangles_h, triangles_v
    else:
        triangles = np.empty((num_triangles, 3), dtype=np.int32)

        # Fill the index buffer (triangles) with indices pointing to the vertex buffer.
        for y in range(height - 1):
            for x in range(width - 1):
                A = y * width + x
                B = y * width + x + 1
                C = (y + 1) * width + x + 1
                D = (y + 1) * width + x
                triangle1 = np.array([A, B, C], dtype=np.int32)
                triangle2 = np.array([A, C, D], dtype=np.int32)
                triangles[(y * (width - 1) + x) * 2] = triangle1
                triangles[(y * (width - 1) + x) * 2 + 1] = triangle2

        # Combine the vertex and index buffers into a mesh.
        return vertices, triangles

def create_indices(width, height):
    indices = np.empty((height, width), dtype=object)
    for y in range(height):
        for x in range(width):
            indices[y, x] = (y, x)
    return indices
def indices_flatten(indices):
    height = indices.shape[0]
    width = indices.shape[1]
    indices_flatten = np.empty((height * width, 2), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            indices_flatten[y * width + x] = indices[y, x]
    return indices_flatten


# TODO: use numpy flatten
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

    indices_changes = indices - concatenated_indices
    return indices_changes


def build_triangle_grid(width, height):
    # padding not implemented
    # TODO: padding? square grid?
    triangles = []
    for i in range(height):
        for j in range(width):
            center_i = i + 0.5
            center_j = j + 0.5
            if (i == height - 1) or (j == width - 1):
                continue
            triangles.append([
                              [(i, j), [center_i, center_j]],
                              [(i + 1, j), [center_i + 1, center_j]],
                              [(i, j + 1), [center_i, center_j + 1]]
                            ])
    return triangles



def create_graph(width, height, vertices, triangles):
    fig, ax = plt.subplots()
    for triangle in triangles:
        poly = Polygon(vertices[triangle], fill=None, edgecolor='black')
        ax.add_patch(poly)

    # Set axis limits
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Set aspect ratio to be equal so that the grid cells aren't distorted
    ax.set_aspect('equal')

    # Show the plot
    plt.show()


def test():
    ## unit test
    width = 5
    height = 2
    indices = create_indices(width, height)
    indices_f = indices_flatten(indices)
    vertices, triangles = create_pixel_center_grid(width, height)

    # tri2 = build_triangle_grid(width, height)
    # print(len(tri2))
    print(vertices.shape)
    print(triangles.shape)
    # create_graph(width, height, vertices, triangles)

# test()