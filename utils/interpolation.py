import numpy as np
import copy

# TODO: HIGH PRIO: boundary condition consideration

"""
one triangle in triangles : 
[
    [(i, j), [center_i, center_j]],
    [(i + 1, j), [center_i + 1, center_j]],
    [(i, j + 1), [center_i, center_j + 1]]
]

"""
import numpy as np
from PIL import Image

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


def sample_bilinear(image, pos_px, height, width):
    x0 = int(pos_px[1])
    y0 = int(pos_px[0])
    x1 = x0
    y1 = y0

    dx = pos_px[1] - float(x0)
    dy = pos_px[0] - float(y0)

    if dx == 0.5 and dy == 0.5:
        return image[y0, x0]

    if dx <= 0.5 and dy <= 0.5:
        x0 -= 1
        y0 -= 1
    elif dx > 0.5 and dy <= 0.5:
        x1 += 1
        y0 -= 1
    elif dx <= 0.5 and dy > 0.5:
        x0 -= 1
        y1 += 1
    elif dx > 0.5 and dy > 0.5:
        x1 += 1
        y1 += 1

    center_p_X = float(x0) + 0.5
    center_p_Y = float(y0) + 0.5

    alpha = abs(pos_px[1] - center_p_X)
    beta = abs(pos_px[0] - center_p_Y)

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    value_A = image[y0, x0]
    value_B = image[y0, x1]
    value_C = image[y1, x0]
    value_D = image[y1, x1]

    value = (1 - alpha) * (1 - beta) * value_A + alpha * (1 - beta) * value_B + (1 - alpha) * beta * value_C + alpha * beta * value_D
    return value

# TODO: on the edge should also be okay
def isPointInTriangle(pt, v1, v2, v3):
    d1 = triangleSign(pt, v1, v2)
    d2 = triangleSign(pt, v2, v3)
    d3 = triangleSign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def triangleSign(pt, a, b):
    return (pt[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (pt[1] - b[1])

def barycentric_coordinates(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01

    bc = np.zeros(3, dtype=float)
    if np.abs(denom) >= 1e-8:
        bc[1] = (d11 * d20 - d01 * d21) / denom
        bc[2] = (d00 * d21 - d01 * d20) / denom
    bc[0] = 1.0 - bc[1] - bc[2]
    return bc

def triangle_interpolation(original_triangles, moved_triangles, orginal_img_changed):
    height = orginal_img_changed.shape[0]
    width = orginal_img_changed.shape[1]
    dst_img = np.copy(orginal_img_changed)
    for idx in range(len(moved_triangles)):
        """
        for each moved triangle, if its 3 vertices are same as the original one, then skip
        """
        if moved_triangles[idx] == original_triangles[idx]:
            continue
        cur_triangle = moved_triangles[idx]
        vert_a = cur_triangle[0][1]
        vert_b = cur_triangle[1][1]
        vert_c = cur_triangle[2][1]

        i_lower = min(vert_a[0], vert_b[0], vert_c[0])
        i_upper = max(vert_a[0], vert_b[0], vert_c[0])

        j_lower = min(vert_a[1], vert_b[1], vert_c[1])
        j_upper = max(vert_a[1], vert_b[1], vert_c[1])

        for i in range(int(i_lower), int(i_upper) + 1):
            for j in range(int(j_lower), int(j_upper) + 1):
                if i < 0 or j < 0 or i >= height or j >= width:
                    continue
                center_pixel = [i + 0.5, j + 0.5]
                if isPointInTriangle(center_pixel, vert_a, vert_b, vert_c):
                    bc = barycentric_coordinates(np.array(center_pixel), np.array(vert_a), np.array(vert_b), np.array(vert_c))
                    src_triangle = original_triangles[idx]
                    src_vert_a = src_triangle[0][1]
                    src_vert_b = src_triangle[1][1]
                    src_vert_c = src_triangle[2][1]
                    src_pt = bc[0] * np.array(src_vert_a) + bc[1] * np.array(src_vert_b) + bc[2] * np.array(src_vert_c)
                    sample_pixel = sample_bilinear(orginal_img_changed, src_pt, height, width)
                    dst_img[i, j] = sample_pixel
    return dst_img


def triangle_interpolation2(triangles, src_vertices, dst_vertices, original_img_changed, resized_img):
    height = resized_img.shape[0]
    width = resized_img.shape[1]
    height_original = original_img_changed.shape[0]
    width_original = original_img_changed.shape[1]
    dst_img = np.copy(original_img_changed)
    for idx in range(len(triangles)):
        cur_triangle = triangles[idx]
        vert_a = dst_vertices[cur_triangle[0]]
        vert_b = dst_vertices[cur_triangle[1]]
        vert_c = dst_vertices[cur_triangle[2]]

        y_lower = min(vert_a[0], vert_b[0], vert_c[0])
        y_upper = max(vert_a[0], vert_b[0], vert_c[0])

        x_lower = min(vert_a[1], vert_b[1], vert_c[1])
        x_upper = max(vert_a[1], vert_b[1], vert_c[1])

        for y in range(int(y_lower), int(y_upper) + 1):
            for x in range(int(x_lower), int(x_upper) + 1):
                if y < 0 or x < 0 or y >= height_original or x >= width_original:
                    continue
                center_pixel = np.array([y + 0.5, x + 0.5])
                if isPointInTriangle(center_pixel, vert_a, vert_b, vert_c):
                    bc = barycentric_coordinates(np.array(center_pixel), np.array(vert_a), np.array(vert_b), np.array(vert_c))
                    src_vert_a = src_vertices[cur_triangle[0]]
                    src_vert_b = src_vertices[cur_triangle[1]]
                    src_vert_c = src_vertices[cur_triangle[2]]
                    src_pt = bc[0] * src_vert_a + bc[1] * src_vert_b + bc[2] * src_vert_c
                    sample_pixel = sample_bilinear(resized_img, src_pt, height, width)
                    dst_img[y, x] = sample_pixel
    return dst_img




def easy_interpolation(orginal_img_changed):
    dst_img = np.copy(orginal_img_changed)

    for i in range(orginal_img_changed.shape[0]):
        pre = orginal_img_changed[i, 0]
        for j in range(orginal_img_changed.shape[1]):
            if j == 0:
                continue
            else:
                if orginal_img_changed[i, j, 0] == 0 and orginal_img_changed[i, j, 1] == 0 and orginal_img_changed[i, j, 2] == 0:
                    dst_img[i, j] = pre
                pre = dst_img[i, j]

    return dst_img








