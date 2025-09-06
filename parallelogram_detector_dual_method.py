# Not stable

import cv2
import numpy as np
import json
import os

# Global debug flag - set to True to enable debug prints and image displays
DEBUG = True
MORPH_KERNEL = (5, 5)  # Kernel size for morphological operations

def load_config():
    """Load configuration from config.json file."""
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config_boxes.json')
    if not os.path.exists(config_path):
        if DEBUG:
            print(f"Configuration file not found at {config_path}. Using the default configuration for parallelogram detection.")
        return {
            "color_ranges": {
                "red": [[[0, 143, 54], [12, 253, 164]], [[162, 143, 54], [179, 253, 164]]], 
                "green": [[[60, 137, 13], [90, 247, 123]]], 
                "blue": [[[94, 173, 45], [124, 255, 155]]], 
                "yellow": [[[7, 170, 99], [37, 255, 209]]]
            },
            "boxes_crop": [0, 0, 600, 600]  # Default crop area [x, y, w, h]
        }
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def isolate_and_subtract_channel(img, channel='r'):
    """
    Channel isolation and subtraction method (V3 preprocessing).
    Isolates a specific channel and subtracts other channels from it.
    """
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


def preprocess_v3_method(img, morph_kernel=MORPH_KERNEL):
    """
    V3 preprocessing: Channel isolation and subtraction.
    Returns processed masks for each RGB channel.
    """
    channels = ['r', 'g', 'b']
    channel_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
    masks = {}
    
    for ch in channels:
        result = isolate_and_subtract_channel(img, ch)
        
        # Extract the relevant channel as a single-channel image
        channel_idx = {"r": 2, "g": 1, "b": 0}[ch]
        single_channel = result[:, :, channel_idx]
        
        # Noise reduction: Apply Gaussian blur before thresholding
        single_channel_blur = cv2.GaussianBlur(single_channel, (5, 5), 0)
        
        # Thresholding
        _, single_channel_thresh = cv2.threshold(single_channel_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        morph_open = cv2.morphologyEx(single_channel_thresh, cv2.MORPH_OPEN, np.ones(morph_kernel, np.uint8))
        closed = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, np.ones(morph_kernel, np.uint8))
        
        masks[ch] = {
            'mask': closed,
            'single_channel_thresh': single_channel_thresh,
            'name': channel_names[ch]
        }
        
        if DEBUG:
            print(f"V3 - {channel_names[ch]} mask shape: {closed.shape}, non-zero pixels: {np.count_nonzero(closed)}")
    
    return masks


def preprocess_v1_2_method(img, color_ranges, morphology_kernel_size=(7, 7), dist_thresh=0.4):
    """
    V1.2 preprocessing: HSV color ranges with distance transform.
    Returns processed data for each color.
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
    
    # Map color names to channel names for consistency
    color_to_channel = {'red': 'r', 'green': 'g', 'blue': 'b'}
    channel_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
    
    data = {}
    for color, ranges in color_ranges.items():
        if color in color_to_channel:  # Only process RGB colors
            ch = color_to_channel[color]
            
            # Create mask for the color
            mask = build_clean_mask(hsv, ranges, morphology_kernel_size)
            
            # Apply distance transform
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, dist_thresh * dist_transform.max(), 255, 0)
            
            data[ch] = {
                'mask': sure_fg.astype(np.uint8),
                'dist_transform': dist_transform,
                'name': channel_names[ch]
            }
            
            if DEBUG:
                print(f"V1.2 - {channel_names[ch]} mask shape: {sure_fg.shape}, non-zero pixels: {np.count_nonzero(sure_fg)}")
    
    return data


def detect_contours_and_centroids(processed_data, img, method='v3', min_area=30, max_aspect_ratio=2.5, show_debug=None):
    """
    Detect contours and find centroids for each channel.
    Works with both V3 and V1.2 preprocessing methods.
    """
    if show_debug is None:
        show_debug = DEBUG
        
    contour_positions = {}
    
    if DEBUG:
        print(f"Processing {len(processed_data)} channels with method {method}")
    
    for ch, data in processed_data.items():
        mask = data['mask']
        name = data['name']
        
        if DEBUG:
            print(f"Processing {name} channel...")
        
        # Find contours
        if method == 'v3':
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        else:  # v1_2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if DEBUG:
            print(f"Found {len(contours)} initial contours in {name}")
        
        if not contours:
            if DEBUG:
                print(f"No contours found in the {name} channel using {method} method.")
            continue
            
        # Debug: Print contour areas
        if show_debug:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                print(f"Contour area in {name} channel ({method}): {area}")
        
        # Filter contours based on area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if DEBUG:
            print(f"After area filtering (min_area={min_area}): {len(contours)} contours in {name}")
        
        if not contours:
            if DEBUG:
                print(f"No contours with sufficient area in {name} channel using {method} method.")
            continue
        
        # Filter contours based on aspect ratio using bounding ellipse
        filtered_contours = []
        for cnt in contours:
            if len(cnt) >= 5:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(cnt)
                # Extract width and height from ellipse (major and minor axes)
                (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                # Calculate aspect ratio using the larger axis as width and smaller as height
                ellipse_width = max(major_axis, minor_axis)
                ellipse_height = min(major_axis, minor_axis)
                if ellipse_height > 0 and ellipse_width / ellipse_height < max_aspect_ratio:
                    filtered_contours.append(cnt)
            else:
                # Fallback to bounding rectangle for contours with less than 5 points
                x, y, w, h = cv2.boundingRect(cnt)
                if h > 0 and w / h < max_aspect_ratio:
                    filtered_contours.append(cnt)
        
        if DEBUG:
            print(f"After aspect ratio filtering (max_ratio={max_aspect_ratio}): {len(filtered_contours)} contours in {name}")
        
        if not filtered_contours:
            if DEBUG:
                print(f"No contours with suitable aspect ratio in {name} channel using {method} method.")
            continue
            
        # Find contour with the highest mean intensity in the relevant channel
        if method == 'v3' and 'single_channel_thresh' in data:
            single_channel_thresh = data['single_channel_thresh']
            brightest_contour = max(filtered_contours, 
                                  key=lambda cnt: cv2.mean(single_channel_thresh, 
                                                         mask=cv2.drawContours(np.zeros_like(single_channel_thresh), [cnt], -1, 255, -1))[0])
        else:
            # For V1.2 method, just take the largest contour
            brightest_contour = max(filtered_contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(brightest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            contour_positions[ch] = (cx, cy)
            
            if show_debug:
                print(f"{name} centroid ({method}): ({cx}, {cy})")
        
        # Display result if debugging
        if show_debug:
            result_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result_bgr, contours, -1, (255, 0, 0), 1)  # Draw all contours in blue
            cv2.drawContours(result_bgr, [brightest_contour], -1, (0, 255, 255), 2)  # Highlight brightest in yellow
            
            # Draw bounding ellipse for the brightest contour
            if len(brightest_contour) >= 5:
                ellipse = cv2.fitEllipse(brightest_contour)
                cv2.ellipse(result_bgr, ellipse, (0, 255, 0), 2)  # Draw ellipse in green
            
            cv2.imshow(f'{name} Channel - {method.upper()} method', result_bgr)
    
    return contour_positions


def find_yellow_contours(img, min_area=100, max_aspect_ratio=2.5):
    """
    Find yellow contours in an image using HSV color detection.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define yellow color range in HSV
    config = load_config()
    lower_yellow = np.array(config['color_ranges']['yellow'][0][0])
    upper_yellow = np.array(config['color_ranges']['yellow'][0][1])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Morphological operations to clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio using bounding ellipse
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    for cnt in contours:
        print(f"Yellow contour area: {cv2.contourArea(cnt)}")
    # Filter by aspect ratio using bounding ellipse
    filtered_contours = []
    for cnt in contours:
        if len(cnt) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(cnt)
            # Extract width and height from ellipse (major and minor axes)
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            # Calculate aspect ratio using the larger axis as width and smaller as height
            ellipse_width = max(major_axis, minor_axis)
            ellipse_height = min(major_axis, minor_axis)
            if ellipse_height > 0 and ellipse_width / ellipse_height < max_aspect_ratio:
                filtered_contours.append(cnt)
        else:
            # Fallback to bounding rectangle for contours with less than 5 points
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 0 and w / h < max_aspect_ratio:
                filtered_contours.append(cnt)
    
    if not filtered_contours:
        return [], mask
        
    return filtered_contours, mask


def calculate_parallelogram_fourth_point(contour_positions, img, search_radius=50, show_debug=None):
    """
    Calculate the fourth point of the parallelogram and find the yellow box there.
    """
    if show_debug is None:
        show_debug = DEBUG
    
    if len(contour_positions) < 3:
        raise ValueError(f"Need at least 3 points to calculate parallelogram, got {len(contour_positions)}")
    
    # Find the two contours with the closest cy values
    channels_with_cy = [(ch, pos[1]) for ch, pos in contour_positions.items()]
    min_diff = float('inf')
    closest_pair = None
    
    for i in range(len(channels_with_cy)):
        for j in range(i + 1, len(channels_with_cy)):
            diff = abs(channels_with_cy[i][1] - channels_with_cy[j][1])
            if diff < min_diff:
                min_diff = diff
                closest_pair = (channels_with_cy[i][0], channels_with_cy[j][0])
    
    if show_debug:
        print(f"The two channels with the closest cy values are: {closest_pair}")
    
    channels = list(contour_positions.keys())
    remaining_contour = [ch for ch in channels if ch not in closest_pair][0]
    
    point_A_color = closest_pair[0]
    point_B_color = closest_pair[1]
    point_C_color = remaining_contour
    
    A = contour_positions[point_A_color]
    B = contour_positions[point_B_color]
    C = contour_positions[point_C_color]
    
    # Calculate two possible fourth points
    D1 = (B[0] + C[0] - A[0], B[1] + C[1] - A[1])  # Opposite A 
    D2 = (A[0] + C[0] - B[0], A[1] + C[1] - B[1])  # Opposite B
    
    if show_debug:
        print(f"Calculated D1: {D1}, D2: {D2}")
    
    # Search for yellow contours around both potential points
    h, w = img.shape[:2]
    yellow_position = None
    
    for D_point, D_name in [(D1, "D1"), (D2, "D2")]:
        # Check if point is within image bounds
        if D_point[0] < 0 or D_point[0] >= w or D_point[1] < 0 or D_point[1] >= h:
            if show_debug:
                print(f"{D_name} is outside image bounds: {D_point}")
            continue
            
        # Create mask for search area
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, D_point, search_radius, 255, -1)
        selected_area = cv2.bitwise_and(img, img, mask=mask)
        
        # Find yellow contours in the selected area
        yellow_contours, yellow_mask = find_yellow_contours(selected_area, min_area=200)
        
        if show_debug:
            print(f"Found {len(yellow_contours)} yellow contours around {D_name} (min_area=50)")
        
        if yellow_contours and len(yellow_contours) > 0:
            if show_debug:
                selected_area_contours = selected_area.copy()
                cv2.drawContours(selected_area_contours, yellow_contours, -1, (255, 0, 0), 2)
                
                # Draw bounding ellipse for yellow contours
                for yellow_cnt in yellow_contours:
                    if len(yellow_cnt) >= 5:
                        ellipse = cv2.fitEllipse(yellow_cnt)
                        cv2.ellipse(selected_area_contours, ellipse, (0, 255, 0), 2)  # Draw ellipse in green
                
                cv2.imshow(f'Yellow Contours around {D_name}', selected_area_contours)
                print(f"Yellow contours found around {D_name}")
            
            # Get centroid of the uppermost yellow contour (lowest y-coordinate)
            # If multiple contours found, select the one with the smallest centroid y-value
            if len(yellow_contours) > 1:
                # Calculate centroids for all contours and find the uppermost one
                contour_centroids = []
                for cnt in yellow_contours:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        contour_centroids.append((cnt, cx, cy))
                
                if contour_centroids:
                    # Sort by y-coordinate (ascending = uppermost first)
                    contour_centroids.sort(key=lambda x: x[2])
                    selected_contour, y_cx, y_cy = contour_centroids[0]
                    
                    if show_debug:
                        print(f"Multiple yellow contours found, selected uppermost at y={y_cy}")
                else:
                    continue  # No valid centroids found
            else:
                # Single contour case
                M = cv2.moments(yellow_contours[0])
                if M['m00'] != 0:
                    y_cx = int(M['m10'] / M['m00'])
                    y_cy = int(M['m01'] / M['m00'])
                else:
                    continue  # Invalid contour
            
            yellow_position = (y_cx, y_cy)
            
            if show_debug:
                result_img = img.copy()
                cv2.circle(result_img, yellow_position, 5, (0, 255, 255), -1)
                cv2.imshow('Yellow contour center found', result_img)
            
            break
    
    if yellow_position is None:
        raise ValueError("Could not find yellow contour at calculated parallelogram positions")
    
    return yellow_position, point_A_color, point_B_color, point_C_color


def identify_corner_arrangement(contour_positions, yellow_position, point_A_color, point_B_color, point_C_color):
    """
    Identify the corner arrangement and return the color string.
    """
    # Create points dictionary
    points = {
        point_A_color: contour_positions[point_A_color], 
        point_B_color: contour_positions[point_B_color], 
        point_C_color: contour_positions[point_C_color], 
        'y': yellow_position
    }
    
    # Sort points by y (vertical position)
    sorted_by_y = sorted(points.items(), key=lambda item: item[1][1])
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    
    # For top points, left/right by x
    top_left = min(top_points, key=lambda item: item[1][0])
    top_right = max(top_points, key=lambda item: item[1][0])
    
    # For bottom points, left/right by x
    bottom_left = min(bottom_points, key=lambda item: item[1][0])
    bottom_right = max(bottom_points, key=lambda item: item[1][0])
    
    # Create result string: top-left, top-right, bottom-right, bottom-left
    corner_sequence = [top_left, top_right, bottom_right, bottom_left]
    result = ",".join([c[0].lower() for c in corner_sequence])
    
    return result, {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }


def detect_parallelogram_dual_method(img_path, max_width=600, max_height=600, method='both', show_debug=None):
    """
    Main function to detect parallelogram using both preprocessing methods.
    
    Args:
        img_path: Path to the input image
        max_width, max_height: Maximum dimensions for resizing
        method: 'v3', 'v1_2', or 'both'
        show_debug: Whether to show debug information and images
    
    Returns:
        Dictionary with results from each method
    """

    # Load and resize image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Resize while keeping aspect ratio
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    results = {}
    
    if method in ['v3', 'both']:
        try:
            if DEBUG:
                print("=== Using V3 Method (Channel Isolation) ===")
            
            # V3 preprocessing
            v3_data = preprocess_v3_method(img_resized)
            
            # Detect contours and centroids
            contour_positions_v3 = detect_contours_and_centroids(v3_data, img_resized, method='v3', show_debug=show_debug)
            
            if len(contour_positions_v3) >= 3:
                # Calculate fourth point and find yellow
                yellow_pos_v3, A_color, B_color, C_color = calculate_parallelogram_fourth_point(
                    contour_positions_v3, img_resized, show_debug=show_debug)
                
                # Identify corner arrangement
                result_v3, corners_v3 = identify_corner_arrangement(
                    contour_positions_v3, yellow_pos_v3, A_color, B_color, C_color)
                
                results['v3'] = {
                    'result_string': result_v3,
                    'contour_positions': contour_positions_v3,
                    'yellow_position': yellow_pos_v3,
                    'corners': corners_v3
                }
                
                if DEBUG:
                    print(f"V3 Result: {result_v3}")
            else:
                if DEBUG:
                    print(f"V3 Method: Insufficient contours found ({len(contour_positions_v3)}/3)")
                results['v3'] = None
                
        except Exception as e:
            if DEBUG:
                print(f"V3 Method failed: {str(e)}")
            results['v3'] = None
    
    if method in ['v1_2', 'both']:
        try:
            if DEBUG:
                print("\n=== Using V1.2 Method (HSV Color Ranges) ===")
            
            # Load config for color ranges
            config_data = load_config()
            
            # V1.2 preprocessing
            v1_2_data = preprocess_v1_2_method(img_resized, config_data['color_ranges'])
            
            # Detect contours and centroids
            contour_positions_v1_2 = detect_contours_and_centroids(v1_2_data, img_resized, method='v1_2', show_debug=show_debug)
            
            if len(contour_positions_v1_2) >= 3:
                # Calculate fourth point and find yellow
                yellow_pos_v1_2, A_color, B_color, C_color = calculate_parallelogram_fourth_point(
                    contour_positions_v1_2, img_resized, show_debug=show_debug)
                
                # Identify corner arrangement
                result_v1_2, corners_v1_2 = identify_corner_arrangement(
                    contour_positions_v1_2, yellow_pos_v1_2, A_color, B_color, C_color)
                
                results['v1_2'] = {
                    'result_string': result_v1_2,
                    'contour_positions': contour_positions_v1_2,
                    'yellow_position': yellow_pos_v1_2,
                    'corners': corners_v1_2
                }
                
                if DEBUG:
                    print(f"V1.2 Result: {result_v1_2}")
            else:
                if DEBUG:
                    print(f"V1.2 Method: Insufficient contours found ({len(contour_positions_v1_2)}/3)")
                results['v1_2'] = None
                
        except Exception as e:
            if DEBUG:
                print(f"V1.2 Method failed: {str(e)}")
            results['v1_2'] = None
    
    return results


def compare_and_decide(results):
    """
    Compare results from both methods and make a decision.
    """
    v3_result = results.get('v3')
    v1_2_result = results.get('v1_2')
    
    # If both methods succeeded
    if v3_result and v1_2_result:
        v3_string = v3_result['result_string']
        v1_2_string = v1_2_result['result_string']
        
        if v3_string == v1_2_string:
            if DEBUG:
                print(f"\nBoth methods agree: {v3_string}")
            return v3_string, "agreement"
        else:
            if DEBUG:
                print(f"\nMethods disagree - V3: {v3_string}, V1.2: {v1_2_string}")
                print("Using V3 result as primary")
            return v3_string, "v3_priority"
    
    # If only one method succeeded
    elif v3_result:
        if DEBUG:
            print(f"\nOnly V3 method succeeded: {v3_result['result_string']}")
        return v3_result['result_string'], "v3_only"
    elif v1_2_result:
        if DEBUG:
            print(f"\nOnly V1.2 method succeeded: {v1_2_result['result_string']}")
        return v1_2_result['result_string'], "v1_2_only"
    else:
        if DEBUG:
            print("\nBoth methods failed")
        return None, "both_failed"


def main():
    """
    Main function demonstrating the dual-method parallelogram detection.
    """
    # Example usage
    img_path = r'C:\Users\efeca\Desktop\new_image.png'  # Update with your image path
    
    try:
        # Run both methods
        results = detect_parallelogram_dual_method(img_path, method='both', show_debug=DEBUG)
        if DEBUG:
            print(results)
        # Compare and decide
        final_result, decision_reason = compare_and_decide(results)
        
        
        print(f"\n=== FINAL RESULT ===")
        print(f"Corner arrangement: {final_result}")
        print(f"Decision reason: {decision_reason}")

        # Only wait for key press and show windows if DEBUG is enabled
        if DEBUG:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {str(e)}")



if __name__ == '__main__':
    main()
