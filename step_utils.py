import os
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import compare_ssim
# from skimage.metrics import structural_similarity as compare_ssim
import re

def image_paths(video_images_path):
    paths = [os.path.join(video_images_path, f) for f in os.listdir(video_images_path) if re.match(r'\d{8}-\w{8}-\d{7}\.jpg', f)]
    paths = sorted(paths, key=os.path.getmtime)
    return paths


def filter2(x, kernel, mode='same'):
    # 窗口内进行高斯卷积，类似加权平均
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    # 高斯加窗
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def img_ssim(prev_frame, curr_frame):
    # convert the images to grayscale
    prev_gray = rgb2gray(prev_frame)
    curr_gray = rgb2gray(curr_frame)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(prev_gray, curr_gray, full=True)
    diff = (diff * 255).astype("uint8")
    # print("SSIM: {}".format(score))

    return score


def calculate_iou(box1, box2):
    '''
   # box1=[top, left, bottom, right]=[x1_min,y1_min,x1_max,y1_max]
   # box2=[top, left, bottom, right]=[x2_min,y2_min,x2_max,y2_max]
    '''
    # in_h = min(x1_max, x2_max) - max(x1_min, x2_min)
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    # in_w = min(y1_max, y2_max) - max(y1_min, y2_min)
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if in_h < 0 or in_w < 0:
        inter = 0
    else:
        inter = in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou
