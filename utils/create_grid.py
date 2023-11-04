import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

def create_grid_strategy_1(width, height):
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

def create_grid_strategy_2(width, height):
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
            if x % 2 == 0:
                if y % 2 == 0:
                    triangle1 = np.array([A, B, C], dtype=np.int32)
                    triangle2 = np.array([A, C, D], dtype=np.int32)
                else:
                    triangle1 = np.array([A, B, D], dtype=np.int32)
                    triangle2 = np.array([B, C, D], dtype=np.int32)
            else:
                if y % 2 == 0:
                    triangle1 = np.array([A, B, D], dtype=np.int32)
                    triangle2 = np.array([B, C, D], dtype=np.int32)
                else:
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



def show_vertices(width, height, vertices, triangles):
    fig, ax = plt.subplots()
    for triangle in triangles:
        poly = Polygon(vertices[triangle], fill=None, edgecolor='black')
        ax.add_patch(poly)

    # Set axis limits
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Set aspect ratio to be equal so that the grid cells aren't distorted
    ax.set_aspect('equal')
    plt.show()


    # create_graph(width, height, vertices, triangles)seam_height

def grid_show():
    # Create a figure and axis
    width = 100
    height = 100
    vertices, triangles = create_grid_strategy_2(width, height)

    x_coords, y_coords = vertices[:, 0], vertices[:, 1]

    # Plot the vertices as points
    plt.scatter(x_coords, y_coords, s=1, color='k')

    # Create a LineCollection for grid lines
    lines = []
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                lines.append([(x_coords[y * width + x], y_coords[y * width + x]),
                              (x_coords[y * width + x + 1], y_coords[y * width + x + 1])])
            if y < height - 1:
                lines.append([(x_coords[y * width + x], y_coords[y * width + x]),
                              (x_coords[(y + 1) * width + x], y_coords[(y + 1) * width + x])])

                # Create a LineCollection from the lines and add it to the plot

    lc = LineCollection(lines, color='k')
    plt.gca().add_collection(lc)

    for triangle_indices in triangles:
        triangle = Polygon([vertices[triangle_indices[0]], vertices[triangle_indices[1]], vertices[triangle_indices[2]],
                            vertices[triangle_indices[0]], ], closed=True, fill=None, edgecolor='k')
        plt.gca().add_patch(triangle)

    plt.grid()

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Invert the Y-axis
    plt.gca().xaxis.set_ticks_position('top')  # Move the X-axis to the top

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Grid Vertices')
    plt.show()


# grid_show()