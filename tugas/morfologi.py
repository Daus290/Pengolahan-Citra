import cv2
import numpy as np
from skimage.morphology import skeletonize

def get_kernel(shape):
    if shape == 'rectangle':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    elif shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    elif shape == 'triangle':
        triangle = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        return triangle
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def apply_morphology(image_path, operation, shape):
    img = cv2.imread(image_path, 0)
    kernel = get_kernel(shape)
    if operation == 'dilate':
        return cv2.dilate(img, kernel, iterations=1)
    else:
        return cv2.erode(img, kernel, iterations=1)

def resize_image(img, max_width=400):
    h, w = img.shape
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def boundary_extraction(img):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    boundary = cv2.subtract(img, eroded)
    return boundary

def convex_hull(img, outline_only=False):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_img = np.zeros_like(img)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        thickness = 2 if outline_only else -1
        cv2.drawContours(hull_img, [hull], -1, 255, thickness)
    return hull_img

def pruning(img):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(img, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated

def region_filling(img):
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    flood_fill = img.copy()
    cv2.floodFill(flood_fill, mask, (0, 0), 255)
    flood_fill_inv = cv2.bitwise_not(flood_fill)
    filled = img | flood_fill_inv
    return filled

def skeletonization(img):
    img = resize_image(img)
    img = img.copy() // 255
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel * 255

def thickening(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.dilate(img, kernel, iterations=1)

def thinning(img):
    from skimage.morphology import skeletonize
    import cv2
    import numpy as np

    img = resize_image(img, max_width=400)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 3)
    binary = binary // 255
    thin = skeletonize(binary).astype(np.uint8) * 255
    return thin

def apply_morphology_lanjutan(image_path, operation, outline_only=False):
    img = cv2.imread(image_path, 0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if operation == 'skeleton':
        binary = resize_image(binary)
    if operation == 'boundary':
        return boundary_extraction(binary)
    elif operation == 'convex':
        return convex_hull(binary, outline_only)
    elif operation == 'pruning':
        return pruning(binary)
    elif operation == 'filling':
        return region_filling(binary)
    elif operation == 'skeleton':
        return skeletonization(binary)
    elif operation == 'thicken':
        return thickening(binary)
    elif operation == 'thinning':
        return thinning(binary)
    elif operation == 'opening':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        return binary
