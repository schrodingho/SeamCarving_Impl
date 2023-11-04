import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

"""
Step 6: Vectorize the pixels, create vertices and triangles
"""
def create_grid_strategy_1(width, height):
    """
    Original Strategy
    A ------- B
    |  *      |
    |    *    |
    |      *  |
    D ------- C
    """
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
    """
    New Strategy
    v ------- v ------- v
    |  *      |      *  |
    |    *    |    *    |
    |      *  |  *      |
    v ------- v ------- v
    |      *  |  *      |
    |    *    |    *    |
    |  *      |      *  |
    v ------- v ------- v
    """
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

def grid_show(vertices, triangles, output_path, figure_size=30, original=False):
    # Plot the grid.
    fig = plt.figure(figsize=(figure_size, figure_size))  # Adjust the figure size as needed
    ax = fig.add_subplot(111)

    x_coords, y_coords = vertices[:, 0], vertices[:, 1]
    # Plot the vertices as points
    ax.scatter(x_coords, y_coords, s=1, color='k')

    # Batch plot triangles using PolyCollection
    poly_verts = [vertices[triangle] for triangle in triangles]
    poly_collection = PolyCollection(poly_verts, edgecolors='black', closed=True, facecolors='white')
    ax.add_collection(poly_collection)

    ax.grid()

    # ax.set_aspect('equal', adjustable='box')
    # ax.set_xlabel('y-axis')
    # ax.set_ylabel('x-axis')
    # # ax.set_title('Grid Vertices')
    # plt.show()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    if original:
        plt.savefig(f"{output_path}/step_6_triangles.png")
        image_grid_show = Image.open(f"{output_path}/step_6_triangles.png")
        image_grid_show_rotated = image_grid_show.rotate(-90)
        image_grid_show_rotated.save(f"{output_path}/step_6_triangles.png")
    else:
        name = "_moved"
        plt.savefig(f"{output_path}/step_7{name}_triangles.png")
        image_grid_show = Image.open(f"{output_path}/step_7{name}_triangles.png")
        image_grid_show_rotated = image_grid_show.rotate(-90)
        image_grid_show_rotated.save(f"{output_path}/step_7{name}_triangles.png")




