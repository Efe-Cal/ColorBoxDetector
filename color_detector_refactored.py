# Channel isolation and HSV range methods with unified contour detection and selection logic

import subprocess
import cv2
import numpy as np
import os
import json


def closeness_to_center(img, detection):
    """Calculate distance from detection center to image center."""
    _, _, cx, cy = detection
    img_cx, img_cy = img.shape[1]//2, img.shape[0]//2
    return np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)

def crop_image(img, x, y, w, h):
    """Crop image to specified region."""
    return img[y:y+h, x:x+w]

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
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
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
    """Determine if a contour is closer to red or yellow color."""
    # Create a mask for the contour
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate the mean HSV values within the contour
    mean_hsv = cv2.mean(hsv, mask=mask)
    mean_h, mean_s, mean_v = mean_hsv[:3]

    red_ranges = [
        (0, 15),    # Lower red range
        (160, 180)  # Upper red range
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


# ===================== PREPROCESSING METHODS =====================

def preprocess_v1_2(img, color_ranges, morphology_kernel_size=(7, 7), dist_thresh=0.4):
    """
    Version 1.2 preprocessing: HSV color ranges with distance transform.
    Returns a dictionary with distance transforms for contour finding.
    """
    def build_clean_mask(hsv, ranges, kernel_size):
        """Build and clean mask for a list of HSV ranges."""
        mask = None
        for lo, hi in ranges:
            part = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        kernel = np.ones(kernel_size, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    data = {}
    for color, ranges in color_ranges.items():
        # Create mask for the color
        mask = build_clean_mask(hsv, ranges, morphology_kernel_size)
        
        # Apply distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, dist_thresh * dist_transform.max(), 255, 0)
        
        data[color] = {
            'dist_transform': dist_transform,
            'thresholded': sure_fg.astype(np.uint8)
        }
    
    return data


def preprocess_v3(img, morph_kernel=(5, 5)):
    """
    Version 3 preprocessing: Channel isolation and subtraction.
    Returns a dictionary of masks for each channel.
    """
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

    channels = ['r', 'g', 'b']
    masks = {}
    
    for ch in channels:
        result = isolate_and_subtract_channel(img, ch)
        
        # Extract the relevant channel as a single-channel image for contour detection
        channel_idx = {"r": 2, "g": 1, "b": 0}[ch]
        single_channel = result[:, :, channel_idx]
        
        _, single_channel_thresh = cv2.threshold(single_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        morph_open = cv2.morphologyEx(single_channel_thresh, cv2.MORPH_OPEN, np.ones(morph_kernel, np.uint8))
        closed = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, np.ones(morph_kernel, np.uint8))
        
        masks[ch] = closed
    
    return masks


# ===================== UNIFIED CONTOUR DETECTION =====================

def detect_contours_v1_2(color_data, min_area_per_contour=4000, min_total_area=500):
    """
    V1.2 contour detection following the original algorithm exactly.
    """
    detections = []
    
    for color, data in color_data.items():
        dist_transform = data['dist_transform']
        
        # Find contours on the distance transform (this is the key difference!)
        contours, _ = cv2.findContours(dist_transform.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate areas for contours > min_area_per_contour
        contour_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area_per_contour]
        
        # Sum all qualifying areas
        total_area = sum(contour_areas)
        if total_area < min_total_area:
            if total_area != 0:
                print(f"Area too small: {total_area}   {color}")
            continue
        
        # Get the largest contour
        largest_idx = np.argmax([cv2.contourArea(cnt) for cnt in contours])
        cnt = contours[largest_idx]
        
        # Calculate centroid
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            detections.append((color, total_area, cx, cy))
    
    return detections


def detect_contours_v3(masks, img, min_area=4000, max_aspect_ratio=2.0, merge_distance=50):
    """
    V3 contour detection following the original algorithm exactly.
    """
    detections = []
    
    for ch, mask in masks.items():
        # Find contours - use RETR_CCOMP as in original
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        # Merge close contours
        contours = merge_close_contours(contours, d_thresh=merge_distance)
        
        # Filter contours based on area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not contours:
            continue
            
        # Filter contours based on aspect ratio
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w / h < max_aspect_ratio:
                filtered_contours.append(cnt)
        
        if not filtered_contours:
            continue
            
        # Sort by area and take the largest
        contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        
        # Calculate centroid and area
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = cv2.contourArea(largest_contour)
            
            # For red channel, determine if it's red or yellow
            if ch == 'r':
                actual_color = is_contour_closer_to_red_or_yellow(img, largest_contour)
                detections.append((actual_color, area, cx, cy))
            else:
                detections.append((ch, area, cx, cy))
    
    return detections


def select_best_detection_v1_2(detections, img):
    """
    V1.2 selection logic following the original algorithm exactly.
    """
    if len(detections) == 0:
        return None
    elif len(detections) == 1:
        return detections[0][0][0]  # Return just the color letter
    else:
        # Normalize area and closeness to center, then combine equally
        areas = np.array([d[1] for d in detections])
        centers = np.array([closeness_to_center(img, d) for d in detections])
        norm_areas = (areas - areas.min()) / (areas.ptp() if areas.ptp() > 0 else 1)
        norm_centers = (centers - centers.min()) / (centers.ptp() if centers.ptp() > 0 else 1)
        scores = norm_areas + (1 - norm_centers)  # larger area and closer to center preferred
        best_idx = np.argmax(scores)
        return detections[best_idx][0][0]  # Return just the color letter


def select_best_detection_v3(detections, img):
    """
    V3 selection logic following the original algorithm exactly, now with area consideration.
    """
    if len(detections) == 0:
        return None
    elif len(detections) == 1:
        return detections[0][0]  # Return just the color
    else:
        # Normalize area and closeness to center, then combine equally (same as v1.2)
        areas = np.array([d[1] for d in detections])
        centers = np.array([closeness_to_center(img, d) for d in detections])
        norm_areas = (areas - areas.min()) / (areas.ptp() if areas.ptp() > 0 else 1)
        norm_centers = (centers - centers.min()) / (centers.ptp() if centers.ptp() > 0 else 1)
        scores = norm_areas + (1 - norm_centers)  # larger area and closer to center preferred
        best_idx = np.argmax(scores)
        return detections[best_idx][0]  # Return just the color


def select_best_detection(detections, img, method='combined'):
    """
    Select the best detection from multiple candidates.
    
    Args:
        detections: List of (color, area, cx, cy) tuples
        img: Original image for center distance calculation
        method: 'area', 'center', 'combined'
    
    Returns:
        Best color detection or None if no detections
    """
    if len(detections) == 0:
        return None
    elif len(detections) == 1:
        return detections[0][0]
    else:
        if method == 'area':
            # Choose largest area
            best_idx = np.argmax([d[1] for d in detections])
        elif method == 'center':
            # Choose closest to center
            distances = [closeness_to_center(img, d) for d in detections]
            best_idx = np.argmin(distances)
        else:  # combined
            # Normalize area and closeness to center, then combine
            areas = np.array([d[1] for d in detections])
            centers = np.array([closeness_to_center(img, d) for d in detections])
            norm_areas = (areas - areas.min()) / (areas.ptp() if areas.ptp() > 0 else 1)
            norm_centers = (centers - centers.min()) / (centers.ptp() if centers.ptp() > 0 else 1)
            scores = norm_areas + (1 - norm_centers)  # larger area and closer to center preferred
            best_idx = np.argmax(scores)
        
        return detections[best_idx][0]


# ===================== MAIN DETECTION FUNCTIONS =====================

def load_config():
    """Load configuration from config.json file."""
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}. Using the hardcoded default configuration.")
        return {
            "big_box_crop": [1735, 657, 172, 122], 
            "color_ranges": {
                "red": [[[0, 143, 54], [12, 253, 164]], [[162, 143, 54], [179, 253, 164]]], 
                "green": [[[60, 137, 13], [90, 247, 123]]], 
                "blue": [[[94, 173, 45], [124, 255, 155]]], 
                "yellow": [[[7, 170, 99], [37, 255, 209]]]
            }
        }
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def extend_color_range(color_ranges, offset=(10, 30, 30)):
    """Extends color ranges by an offset."""
    extended_ranges = {}
    for color, ranges in color_ranges.items():
        extended_ranges[color] = []
        for color_range in ranges:
            lo, hi = color_range
            lo_extended = [max(0, c - o) for c, o in zip(lo, offset)]
            hi_extended = [min(255, c + o) for c, o in zip(hi, offset)]
            extended_ranges[color].append([lo_extended, hi_extended])
    return extended_ranges


def detect_color_v1_2(img_path, retry_with_extended=True):
    """
    Version 1.2 detection using HSV color ranges and distance transform.
    """
    def _detect_with_ranges(color_ranges, is_retry=False):
        config = load_config()
        image = cv2.imread(img_path)
        box_image = crop_image(image, *config["big_box_crop"])
        
        # Preprocess
        color_data = preprocess_v1_2(box_image, color_ranges)
        
        # Detect contours
        detections = detect_contours_v1_2(color_data)
        
        # Select best
        result = select_best_detection_v1_2(detections, box_image)
        
        if result is None and not is_retry and retry_with_extended:
            print("No boxes detected, trying with extended color ranges...")
            extended_ranges = extend_color_range(color_ranges)
            return _detect_with_ranges(extended_ranges, is_retry=True)
        elif result is None and is_retry:
            print("No boxes detected even after extending ranges")
        
        return result
    
    config = load_config()
    return _detect_with_ranges(config["color_ranges"])


def detect_color_v3(img_path):
    """
    Version 3 detection using channel isolation and subtraction.
    """
    config = load_config()
    image = cv2.imread(img_path)
    box_image = crop_image(image, *config["big_box_crop"])
    
    # Preprocess using channel isolation
    masks = preprocess_v3(box_image)
    
    # Detect contours
    detections = detect_contours_v3(masks, box_image, min_area=4000, max_aspect_ratio=2.0, merge_distance=50)
    
    # Select best
    result = select_best_detection_v3(detections, box_image)
    
    if result is None:
        print("Nothing detected, defaulting to blue")
        return None
    
    return result


def decider(v1_2, v3):
    """Decision logic for combining results from both methods."""
    if v1_2 == v3:
        return v1_2
    if v1_2 in ["r", "y"] and v3 in ["r", "y"]:
        return v1_2
    if v3 == "g":
        return 'g'
    if v1_2 is None:
        return v3
    if v3 is None:
        return v1_2
    if v1_2 is None and v3 is None:
        return "b"
    
    # Default fallback
    return v1_2


def image_job(img_path="./image.png"):
    """
    Main function to capture image and detect color using both methods.
    """
    # Capture image using rpicam-still
    subprocess.run([
        "rpicam-still", "--output", img_path, "--timeout", "200", 
        "--width", "1920", "--height", "1080", "--rotation", "180"
    ])
    
    # Run both detection methods
    v1_2 = detect_color_v1_2(img_path)
    v3 = detect_color_v3(img_path)
    
    print(f"v1.2 detected: {v1_2}, v3 detected: {v3}")
    
    # Use decision logic to combine results
    final_result = decider(v1_2, v3)
    return final_result



# ===================== LEGACY FUNCTION COMPATIBILITY =====================

def get_box_color(img_path):
    """Legacy function for backward compatibility."""
    return detect_color_v1_2(img_path)


def detect_and_extract_contours(img_path):
    """Legacy function for backward compatibility."""
    return detect_color_v3(img_path)


def main():
    import glob
    for img_path in glob.glob(r"C:\Users\efeca\Desktop\imgs\*.png"):
        print(f"Processing {img_path}")
        
        v1_2 = detect_color_v1_2(img_path)
        v3 = detect_color_v3(img_path)
        
        print(f"v1.2 detected: {v1_2}, v3 detected: {v3}")
        
        # Use decision logic to combine results
        final_result = decider(v1_2, v3)
        print(f"Final detected color: {final_result}")

if __name__ == '__main__':
    main()