# Detection of colored corners of a parallelogram using HSV range masking

import cv2
import numpy as np
import json
import os

DEBUG = True

# Default morphology kernel sizes
MORPH_KERNEL_COLOR = (7, 7)
MORPH_KERNEL_YELLOW = (5, 5)


def load_config():
    """Load configuration JSON or return defaults if missing."""
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config_boxes.json')
    if not os.path.exists(config_path):
        if DEBUG:
            print(f"Config not found at {config_path}, using defaults.")
        return {
            "color_ranges": {
                "red": [[[0, 143, 54], [12, 253, 164]], [[162, 143, 54], [179, 253, 164]]],
                "green": [[[60, 137, 13], [90, 247, 123]]],
                "blue": [[[94, 173, 45], [124, 255, 155]]],
                "yellow": [[[7, 170, 99], [37, 255, 209]]]
            },
            "boxes_crop": [0, 0, 600, 600]
        }
    with open(config_path, 'r') as f:
        return json.load(f)


def preprocess_v1_2_method(img, color_ranges, morphology_kernel_size=MORPH_KERNEL_COLOR, dist_thresh=0.4):
    """V1.2 preprocessing: HSV color masks + distance transform to tighten foreground.
    Returns dict keyed by channel letter ('r','g','b') containing: mask, dist_transform, name.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def build_clean_mask(ranges):
        mask_acc = None
        for lo, hi in ranges:
            part = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask_acc = part if mask_acc is None else cv2.bitwise_or(mask_acc, part)
        kernel = np.ones(morphology_kernel_size, np.uint8)
        mask_acc = cv2.morphologyEx(mask_acc, cv2.MORPH_OPEN, kernel)
        mask_acc = cv2.morphologyEx(mask_acc, cv2.MORPH_CLOSE, kernel)
        return mask_acc

    color_to_channel = {'red': 'r', 'green': 'g', 'blue': 'b'}
    channel_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}

    data = {}
    for color, ranges in color_ranges.items():
        if color not in color_to_channel:
            continue  # skip yellow here
        ch = color_to_channel[color]
        mask = build_clean_mask(ranges)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, dist_thresh * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        data[ch] = {
            'mask': sure_fg,
            'dist_transform': dist_transform,
            'name': channel_names[ch]
        }
        if DEBUG:
            print(f"V1.2 preprocess {channel_names[ch]}: non-zero={np.count_nonzero(sure_fg)}")
    return data


def detect_contours_and_centroids(processed_data, img, min_area=30, max_aspect_ratio=2.5):
    """Detect contours and pick one representative centroid per channel using V1.2 masks."""
    contour_positions = {}
    if DEBUG:
        print(f"Detecting contours for {len(processed_data)} channels (V1.2)")

    for ch, data in processed_data.items():
        mask = data['mask']
        name = data['name']
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if DEBUG:
            print(f"{name}: initial contours={len(contours)}")
        if not contours:
            continue
        # filter by area
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not contours:
            if DEBUG: print(f"{name}: all contours below area {min_area}")
            continue
        # aspect ratio filter
        filtered = []
        for cnt in contours:
            if len(cnt) >= 5:
                (cx, cy), (maj, minr), ang = cv2.fitEllipse(cnt)
                w = max(maj, minr); h = min(maj, minr)
                # print aspect ratio
                if cv2.contourArea(cnt) < 200:
                    max_aspect_ratio = 3.7
                print(f"{name} contour aspect ratio: {w/h:.2f}")
                if h > 0 and w / h < max_aspect_ratio:
                    filtered.append(cnt)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > 0 and w / h < max_aspect_ratio:
                    filtered.append(cnt)
        if not filtered:
            if DEBUG: print(f"{name}: removed by aspect ratio filter")
            continue
        # choose largest area
        best = max(filtered, key=cv2.contourArea)
        M = cv2.moments(best)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
        contour_positions[ch] = (cx, cy)
        if DEBUG:
            print(f"{name} centroid: ({cx},{cy})")
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, filtered, -1, (255,0,0), 1)
            cv2.drawContours(vis, [best], -1, (0,255,255), 2)
            if len(best) >= 5:
                ellipse = cv2.fitEllipse(best)
                cv2.ellipse(vis, ellipse, (0,255,0), 2)
            cv2.imshow(f'{name} channel mask', vis)
    return contour_positions


def find_yellow_contours(img, color_ranges, min_area=200, max_aspect_ratio=2.5):
    """Detect yellow contours using HSV ranges from config."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_ranges = color_ranges['yellow']
    mask = None
    for lo, hi in yellow_ranges:
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_KERNEL_YELLOW, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(MORPH_KERNEL_YELLOW, np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    filtered = []
    for c in contours:
        if len(c) >= 5:
            (_, _), (maj, minr), _ = cv2.fitEllipse(c)
            w = max(maj, minr); h = min(maj, minr)
            print(f"Yellow contour aspect ratio: {w/h:.2f}")
            if cv2.contourArea(c) < 200:
                    max_aspect_ratio = 3.7
            if h > 0 and w / h < max_aspect_ratio:
                filtered.append(c)
        else:
            x,y,w,h = cv2.boundingRect(c)
            if h>0 and w/h < max_aspect_ratio:
                filtered.append(c)
    return filtered, mask


def calculate_parallelogram_fourth_point(contour_positions, img, color_ranges, search_radius=50):
    """Given 3 colored corner centroids (r,g,b) infer yellow corner and locate it in image."""
    if len(contour_positions) < 3:
        raise ValueError(f"Need at least 3 points, got {len(contour_positions)}")
    # determine closest pair in vertical (y) to identify A,B vs remaining C
    channels_with_cy = list(contour_positions.items())  # [(ch,(x,y)),...]
    min_diff = float('inf'); closest_pair = None
    for i in range(len(channels_with_cy)):
        for j in range(i+1, len(channels_with_cy)):
            diff = abs(channels_with_cy[i][1][1] - channels_with_cy[j][1][1])
            if diff < min_diff:
                min_diff = diff
                closest_pair = (channels_with_cy[i][0], channels_with_cy[j][0])
    remaining = [ch for ch in contour_positions.keys() if ch not in closest_pair][0]
    A_color, B_color = closest_pair
    C_color = remaining
    A = contour_positions[A_color]; B = contour_positions[B_color]; C = contour_positions[C_color]
    D_candidates = [ (B[0] + C[0] - A[0], B[1] + C[1] - A[1]),  # D1 opposite A
                      (A[0] + C[0] - B[0], A[1] + C[1] - B[1]) ] # D2 opposite B
    h, w = img.shape[:2]
    yellow_position = None
    for idx, D_point in enumerate(D_candidates, start=1):
        x,y = D_point
        if x < 0 or x >= w or y < 0 or y >= h:
            if DEBUG: print(f"D{idx} out of bounds {D_point}")
            continue
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, D_point, search_radius, 255, -1)
        selected = cv2.bitwise_and(img, img, mask=mask)
        yellow_contours, yellow_mask = find_yellow_contours(selected, color_ranges, min_area=100)
        if DEBUG:
            print(f"D{idx}: yellow contours found={len(yellow_contours)}")
        if yellow_contours:
            # choose upper-most centroid (smallest y)
            centroids = []
            for c in yellow_contours:
                M = cv2.moments(c)
                if M['m00'] == 0: continue
                cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                centroids.append((cx, cy, c))
            # draw all contours and centroids for debug
            if DEBUG:
                vis = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(vis, yellow_contours, -1, (255,0,0), 1)
                for cx,cy,_ in centroids:
                    cv2.circle(vis, (cx,cy), 3, (0,255,255), -1)
                cv2.circle(vis, D_point, search_radius, (0,0,255), 1)
                cv2.imshow(f'Yellow search around D{idx}', vis)
            
            if not centroids:
                continue
            centroids.sort(key=lambda t: t[1])
            y_cx, y_cy, chosen_contour = centroids[0]
            yellow_position = (y_cx, y_cy)
            if DEBUG:
                vis = img.copy()
                cv2.circle(vis, yellow_position, 5, (0,255,255), -1)
                cv2.imshow(f'Yellow around D{idx}', vis)
            break
    if yellow_position is None:
        raise ValueError("Yellow corner not found at predicted positions")
    return yellow_position, A_color, B_color, C_color


def identify_corner_arrangement(contour_positions, yellow_position, A_color, B_color, C_color):
    """Return corner order string and mapping of named corners."""
    points = {
        A_color: contour_positions[A_color],
        B_color: contour_positions[B_color],
        C_color: contour_positions[C_color],
        'y': yellow_position
    }
    sorted_by_y = sorted(points.items(), key=lambda kv: kv[1][1])
    top = sorted_by_y[:2]; bottom = sorted_by_y[2:]
    top_left = min(top, key=lambda kv: kv[1][0])
    top_right = max(top, key=lambda kv: kv[1][0])
    bottom_left = min(bottom, key=lambda kv: kv[1][0])
    bottom_right = max(bottom, key=lambda kv: kv[1][0])
    order = [top_left, top_right, bottom_right, bottom_left]
    result = ",".join([c[0] for c in order])
    return result, {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }


def detect_parallelogram_v1_2(img_path, max_width=600, max_height=600, show_debug=None):
    """High-level detection using only V1.2 method.
    Returns dict with result_string, contour_positions, yellow_position, corners.
    """
    if show_debug is None:
        show_debug = DEBUG
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))

    config = load_config()
    data = preprocess_v1_2_method(img_resized, config['color_ranges'])
    contour_positions = detect_contours_and_centroids(data, img_resized, max_aspect_ratio=2.5)
    if len(contour_positions) < 3:
        cv2.waitKey(0)
        raise ValueError(f"Insufficient colored corners detected ({len(contour_positions)}/3)")
    yellow_pos, A_color, B_color, C_color = calculate_parallelogram_fourth_point(contour_positions, img_resized, config['color_ranges'])
    result_string, corners = identify_corner_arrangement(contour_positions, yellow_pos, A_color, B_color, C_color)
    if show_debug:
        print(f"Result: {result_string}")
    return {
        'result_string': result_string,
        'contour_positions': contour_positions,
        'yellow_position': yellow_pos,
        'corners': corners
    }


def main():
    # Example usage (replace path)
    import glob
    for img_path in glob.glob(r"C:\Users\efeca\Desktop\np\*.png"):
        try:
            result = detect_parallelogram_v1_2(img_path, show_debug=DEBUG)
            print("=== FINAL V1.2 RESULT ===")
            print(f"Corner arrangement: {result['result_string']}")
            if DEBUG:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
