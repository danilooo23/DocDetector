import cv2 as cv
import numpy as np
import os
import sys

# usado solo para debuggear
def debug_image(img, name, debug_dir="output"):
    os.makedirs(debug_dir, exist_ok=True)
    cv.imwrite(os.path.join(debug_dir, f"{name}.jpg"), img)

def clean_background_by_contour(binary):
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(binary) * 255

    largest = max(contours, key=cv.contourArea)
    mask = np.ones_like(binary) * 255
    cv.drawContours(mask, [largest], -1, 0, 2)

    return mask

def detect_harris_points(gray_img):
    gray_float = np.float32(gray_img)
    harris_response = cv.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    harris_dilated = cv.dilate(harris_response, None)

    threshold = 0.01 * harris_dilated.max()
    points = np.argwhere(harris_dilated > threshold)

    keypoints = [cv.KeyPoint(x=float(x), y=float(y), size=3) for y, x in points]

    return keypoints

def draw_keypoints(image, keypoints):
    output = image.copy()
    if keypoints is not None:
        for p in keypoints:
            x, y = map(int, p.pt)
            cv.circle(output, (x, y), 3, (0, 0, 255), -1)
    return output

def preprocess_image(image, i):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, 10)

    cleaned_binary = clean_background_by_contour(binary)

    return cleaned_binary

def load_image(path):
    image = cv.imread(path)
    if image is None:
        print(f"Could not load image '{path}'.")
    return image

def split_quadrants(keypoints, descriptors, shape):
    h, w = shape
    center_x, center_y = w // 2, h // 2

    quadrants = {
        "sup_izq": [],
        "sup_der": [],
        "inf_izq": [],
        "inf_der": []
    }

    for kp, desc in zip(keypoints, descriptors):
        x, y = kp.pt
        if x < center_x and y < center_y:
            quadrants["sup_izq"].append((kp, desc))
        elif x >= center_x and y < center_y:
            quadrants["sup_der"].append((kp, desc))
        elif x < center_x and y >= center_y:
            quadrants["inf_izq"].append((kp, desc))
        else:
            quadrants["inf_der"].append((kp, desc))

    return quadrants

def find_best_points(quadrants, ref_quads, threshold=40):
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    best_points = []

    for name, points in quadrants.items():
        if not points:
            continue

        kps, descs = zip(*points)
        descs = np.array(descs)
        ref_descs = ref_quads[name]

        if len(ref_descs) == 0:
            print(f"⚠️ No reference descriptors for quadrant: {name}")
            continue

        matches = bf.knnMatch(descs, ref_descs, k=1)
        distances = [m[0].distance for m in matches]
        min_idx = np.argmin(distances)

        if distances[min_idx] < threshold:
            best_points.append(kps[min_idx])

    return best_points

def extract_brief_descriptors(image, keypoints):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    return gray, *brief.compute(gray, keypoints)

def draw_best_points(image, points):
    img = image.copy()
    for kp in points:
        x, y = map(int, kp.pt)
        cv.circle(img, (x, y), 15, (0, 255, 255), -1)
    return img

def order_points(points):
    points = np.array(points, dtype="float32")

    sum_coords = points.sum(axis=1)
    diff_coords = np.diff(points, axis=1)

    sup_izq = points[np.argmin(sum_coords)]
    inf_der = points[np.argmax(sum_coords)]
    sup_der = points[np.argmin(diff_coords)]
    inf_izq = points[np.argmax(diff_coords)]

    return np.array([sup_izq, sup_der, inf_der, inf_izq], dtype="float32")

def rectify_image(image, corners):
    ordered_corners = order_points(corners)

    (tl, tr, br, bl) = ordered_corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    max_width = int(max(widthA, widthB))

    a4_ratio = 297 / 210
    max_height = int(max_width * a4_ratio)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(ordered_corners, dst)
    warped = cv.warpPerspective(image, M, (max_width, max_height))

    return warped

def scanner(image, ref_quads_path="reference_descriptors_by_quadrant.npz"):
    ref_quads = np.load(ref_quads_path)

    preprocessed_img = preprocess_image(image, 0)
    keypoints = detect_harris_points(preprocessed_img)
    gray, keypoints, descriptors = extract_brief_descriptors(image, keypoints)
    quadrants = split_quadrants(keypoints, descriptors, gray.shape)
    best_points = find_best_points(quadrants, ref_quads)

    if len(best_points) == 4:
        points = [kp.pt for kp in best_points]
        final_img = rectify_image(image, points)
        return final_img
    else:
        return None


