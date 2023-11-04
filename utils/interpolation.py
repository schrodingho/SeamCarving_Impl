import numpy as np

"""
Step 8: Interpolation
Mainly follow the method and code from Assignment 3 (Image Warping)
"""
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

def triangle_interpolation(triangles, src_vertices, dst_vertices, original_img_changed, resized_img):
    print("*Interpolation Running (may take longer time)...")
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







