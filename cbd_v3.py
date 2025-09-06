# Channel isolation method with contour merging

import subprocess
import cv2
import numpy as np

MORPH_KERNEL = (5, 5)  # Kernel size for morphological operations

def isolate_and_subtract_channel(img, channel='r'):
    
    if img is None:
        raise ValueError("Image not found or path is incorrect.")

    channels = {'b': 0, 'g': 1, 'r': 2}
    if channel not in channels:
        raise ValueError("Channel must be 'r', 'g', or 'b'.")

    idx = channels[channel]
    other_idxs = [i for i in range(3) if i != idx]

    # Subtract other channels from the selected channel
    result_channel = img[:, :, idx]
    for oi in other_idxs:
        result_channel = cv2.subtract(result_channel, img[:, :, oi])

    # Create output image with only the result in the selected channel
    out_img = np.zeros_like(img)
    out_img[:, :, idx] = result_channel
    return out_img


# Example usage:
channels = ['r', 'g', 'b']
channel_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
img_path = './image.png'

def crop_image(img, x, y, w, h):
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

def merge_close_contours(contours, d_thresh=20):
    """
    Merge contours whose minimum point-to-point distance <= d_thresh (pixels).
    Returns a list of merged contours (convex hulls).
    """
    if not contours:
        return []

    n = len(contours)
    rects = [cv2.boundingRect(c) for c in contours]  # (x,y,w,h)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    def bbox_dist(r1, r2):
        x1,y1,w1,h1 = r1
        x2,y2,w2,h2 = r2
        x_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        y_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        return (x_gap**2 + y_gap**2)**0.5

    def contour_min_dist(c1, c2):
        p1 = c1.reshape(-1, 2).astype(np.float32)
        p2 = c2.reshape(-1, 2).astype(np.float32)
        # vectorized pairwise distances
        d = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(axis=2))
        return float(d.min())

    # build connectivity (fast bbox filter, then exact distance)
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_dist(rects[i], rects[j]) > d_thresh:
                continue
            if contour_min_dist(contours[i], contours[j]) <= d_thresh:
                union(i, j)

    # group and merge
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged = []
    for idxs in groups.values():
        pts = np.vstack([contours[k].reshape(-1, 2) for k in idxs])
        hull = cv2.convexHull(pts.astype(np.int32))
        merged.append(hull)

    return merged

def is_contour_closer_to_red_or_yellow(img, contour):

    # Create a mask for the contour
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate the mean HSV values within the contour
    mean_hsv = cv2.mean(hsv, mask=mask)
    mean_h, mean_s, mean_v = mean_hsv[:3]

    red_ranges = [
        (0, 10),    # Lower red range
        (170, 180)  # Upper red range
    ]
    yellow_range = (20, 30)
    
    # Calculate distance to red (considering wraparound)
    red_distance = float('inf')
    for red_min, red_max in red_ranges:
        if red_min <= mean_h <= red_max:
            red_distance = 0  # Direct hit
            break
        else:
            # Calculate minimum distance to this range
            dist_to_range = min(abs(mean_h - red_min), abs(mean_h - red_max))
            red_distance = min(red_distance, dist_to_range)
    
    # Handle hue wraparound for red (0-180 scale)
    if mean_h > 90:  # If hue is in upper half, also check distance to 0
        wraparound_distance = min(abs(mean_h - 180), abs(mean_h - 0))
        if wraparound_distance < 10:  # Within red range considering wraparound
            red_distance = min(red_distance, wraparound_distance)
    
    # Calculate distance to yellow
    yellow_min, yellow_max = yellow_range
    if yellow_min <= mean_h <= yellow_max:
        yellow_distance = 0
    else:
        yellow_distance = min(abs(mean_h - yellow_min), abs(mean_h - yellow_max))
    
    # Return the closer color
    if red_distance <= yellow_distance:
        return 'r'
    else:
        return 'y'

def detect_and_extract_contours(img_path):
    img = cv2.imread(img_path)
    img = crop_image(img, 1735, 657, 172, 122)
    cv2.imshow('Cropped Image', img)

    for ch in channels:
        result = isolate_and_subtract_channel(img, ch)

        # Extract the relevant channel as a single-channel image for contour detection
        channel_idx = {"r": 2, "g": 1, "b": 0}[ch]
        single_channel = result[:, :, channel_idx]
        cv2.imshow(f'{channel_names[ch]} Channel', single_channel)

        _, single_channel_thresh = cv2.threshold(single_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        morph_open = cv2.morphologyEx(single_channel_thresh, cv2.MORPH_OPEN, np.ones(MORPH_KERNEL, np.uint8))
        closed = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, np.ones(MORPH_KERNEL, np.uint8))

        # Convert closed image to BGR for colored drawing
        result_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        contours = merge_close_contours(contours, d_thresh=50)
        print(f"{channel_names[ch]} channel - Contours after merging: {len(contours)}")
        
        cv2.drawContours(result_bgr, contours, -1, (255, 0, 0), 1)  # Draw contours in blue
        
        # Debug: Print contour areas
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     print(f"Contour area in {channel_names[ch]} channel: {area}")

        # Filter contours based on area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 4000]
        print(f"{channel_names[ch]} channel - Contours after area filtering: {len(contours)}")
        # Filter contours based on aspect ratio
        ratio = 2
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] < ratio]

        cv2.drawContours(result_bgr, contours, -1, (0, 255, 255), 2)  # Highlight brightest contour in yellow
        cv2.imshow(f'Isolated {channel_names[ch]} Channel', result_bgr)
        
        if len(contours) > 0 and ch=="r":
            color = is_contour_closer_to_red_or_yellow(img, contours[0])
            print(f"Detected color: {color}")
            return color
        if len(contours)>0:
            return ch
        
    print("Nothing detected, defaulting to blue")
    return "b"


if __name__ == '__main__':
    image_path = input("Enter the path to the image file: ").strip()
    print(detect_and_extract_contours(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        