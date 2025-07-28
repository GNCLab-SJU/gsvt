import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    # r = 60
    r = 30
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t

def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * im + mean_b
    return q

def BrightChannel(im,sz):
    b,g,r = cv2.split(im)
    bc = cv2.max(cv2.max(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    bright = cv2.dilate(bc,kernel)
    return bright

def sobel_edge_extractor(img):
    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    absx= cv2.convertScaleAbs(sobel_horizontal)
    absy = cv2.convertScaleAbs(sobel_vertical)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return edge

def normalize_img(img):
    return (img - img.min()) / (img.max() - img.min())

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def get_atmosphere_w_location_drop(I, darkch, p, drop_rate):
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[ int(M*N*drop_rate): int(M*N*drop_rate + M*N*p)]  # find top M * N * p indexes
    collected_pixels = flatI.take(searchidx, axis=0)
    avg_pixels = np.average(collected_pixels, axis=1)
    max_avg_pixel_id = np.argmax(avg_pixels)
    location_in_flat = searchidx[max_avg_pixel_id]
    y = (location_in_flat)//N + 1
    x = (location_in_flat)%N + 1
    return collected_pixels[max_avg_pixel_id], x, y

def move_points_away_from_anchor(anchor, points, distances):
    """
    Move each point away from the anchor by a corresponding distance.

    Parameters:
    - anchor: Tuple of (x, y), the anchor point
    - points: List of 10 tuples [(x1, y1), ..., (x10, y10)]
    - distances: List of 10 distances [d1, ..., d10]

    Returns:
    - new_points: List of 10 new points moved away from anchor
    """
    anchor = np.array(anchor)
    new_points = []

    for point, dist in zip(points, distances):
        point = np.array(point)
        direction = point - anchor
        norm = np.linalg.norm(direction)

        if norm == 0:
            # If the point is exactly at the anchor, move in arbitrary direction (e.g., along x-axis)
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm  # Normalize

        new_point = point + direction * dist  # Move away from anchor
        new_points.append(tuple(new_point))

    return new_points


def move_points_away_from_anchor_vectorized(anchor, points, distances):
    """
    Vectorized version: move points away from the anchor by given distances.
    
    Parameters:
    - anchor: Tuple (x, y)
    - points: List of shape (N, 2)
    - distances: List of N distance values

    Returns:
    - new_points: List of shape (N, 2)
    """
    anchor = np.array(anchor)
    points = np.array(points)
    distances = np.array(distances)

    # Direction vectors from anchor to points
    directions = points - anchor  # shape (N, 2)

    # Normalize the directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)  # shape (N, 1)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    unit_directions = directions / norms

    # Multiply each unit direction by the corresponding distance
    displacements = unit_directions * distances[:, np.newaxis]

    # New points
    new_points = points + displacements

    return new_points.tolist()



path = 'test_imgs/6_hazy.jpg'
fname = get_file_name(path)
print(fname)

hazy_img = cv2.imread(path)
hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
print(hazy_img.shape)

src = hazy_img.copy()
I = src.astype("float64") / 255

dark2 = DarkChannel(I, 3)
t2 = TransmissionRefine(src, dark2)

bright1 = BrightChannel(I, 30)
tb1 = TransmissionRefine(src, bright1)

a = b = 3
multiplied = (t2**a)*(tb1**b)
mul1_clip = multiplied.copy()
mul1_clip[np.where(mul1_clip < 0.01)] = 0

gray = cv2.cvtColor(hazy_img, cv2.COLOR_RGB2GRAY)
sobel = 1 - sobel_edge_extractor(gray)/255
multi = 2*mul1_clip*sobel
multi = np.clip(multi, 0.0, 1.0)

multi_mean = np.mean(multi)*255

multi_norm = normalize_img(multi)
test_t = 1 - multi_norm

offset = 0.1
test_t_new = (1-offset)*test_t + offset
test_t = test_t_new

hazy_hsv = cv2.cvtColor(hazy_img, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(hazy_hsv)

A2, Ax2, Ay2 = get_atmosphere_w_location_drop(I, multi, 0.001, 0.00)

A2_8bit = hazy_img.copy()
A2_8bit[:,:,0] = int(A2[0]*255)
A2_8bit[:,:,1] = int(A2[1]*255)
A2_8bit[:,:,2] = int(A2[2]*255)
A2_8bit_to_hsv = cv2.cvtColor(A2_8bit, cv2.COLOR_RGB2HSV)
A2_8bit_hsv_values = np.mean(A2_8bit_to_hsv, axis=0)[0]
# print(A2_8bit_hsv_values)

anchor_point = (int(A2_8bit_hsv_values[1]), int(A2_8bit_hsv_values[2]))
# print(anchor_point)

s_flatten = s.ravel()
v_flatten = v.ravel()
t_flatten = test_t.ravel()

list_sv = np.array([s_flatten,v_flatten])
list_sv_reshaped = list(zip(*list_sv))
list_sv_reshaped = np.array(list_sv_reshaped)


distances = t_flatten*multi_mean
new_positions = move_points_away_from_anchor_vectorized(anchor_point, list_sv_reshaped, distances)

new_positions = np.array(new_positions)

new_s = new_positions[:,0]
new_v = new_positions[:,1]

new_s_rs = np.reshape(new_s, s.shape)
new_v_rs = np.reshape(new_v, v.shape)

enh_s = np.clip(new_s_rs, 0, 255).astype('uint8')
enh_v = np.clip(new_v_rs, 0, 255).astype('uint8')

new_hsv = cv2.merge([h,enh_s,enh_v])
rgb_back = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)


# plt.figure(figsize = (18, 9))
plt.subplot(121), plt.imshow(hazy_img), plt.title("Hazy"), plt.axis('off')
plt.subplot(122), plt.imshow(rgb_back), plt.title("Output"), plt.axis('off')
plt.show()