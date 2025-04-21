import argparse
import sys
import cv2
import numpy as np
import os
import json


def select_color_region(image_path):
    """
    Opens an image and allows the user to select a rectangular region.
    Returns the average HSV value of the selected area.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (h, s, v) average values of the selected region
        
    Usage:
        Press and drag to select a region
        Press 'c' to confirm selection
        Press 'r' to reset selection
        Press 'q' to quit without selection
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image at {image_path}")
        return None
    
    # Create window and set mouse callback
    window_name = "Select Color Region"
    cv2.namedWindow(window_name)
    
    # Variables to store rectangle coordinates
    rect_start = None
    rect_end = None
    dragging = False
    selection_complete = False
    
    # Clone image for drawing
    img_copy = img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal rect_start, rect_end, dragging, img_copy, selection_complete
        
        # Reset the image copy if needed
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_start = (x, y)
            rect_end = (x, y)
            dragging = True
            img_copy = img.copy()
            selection_complete = False
            
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            rect_end = (x, y)
            # Draw rectangle on the image copy
            temp_img = img.copy()
            cv2.rectangle(temp_img, rect_start, rect_end, (0, 255, 0), 2)
            img_copy = temp_img
            
        elif event == cv2.EVENT_LBUTTONUP:
            rect_end = (x, y)
            dragging = False
            # Draw the final rectangle
            cv2.rectangle(img_copy, rect_start, rect_end, (0, 255, 0), 2)
            selection_complete = True
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Main loop
    while True:
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF
        
        # Confirm selection
        if key == ord('c') and selection_complete:
            break
        
        # Reset selection
        elif key == ord('r'):
            rect_start = None
            rect_end = None
            img_copy = img.copy()
            selection_complete = False
            
        # Quit
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    
    # If we have a valid selection
    if rect_start and rect_end:
        # Ensure correct order of coordinates (top-left, bottom-right)
        x1, y1 = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
        x2, y2 = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])
        
        # Extract the region
        region = img[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Calculate average HSV values
        h_avg = int(np.mean(hsv_region[:, :, 0]))
        s_avg = int(np.mean(hsv_region[:, :, 1]))
        v_avg = int(np.mean(hsv_region[:, :, 2]))
        
        # Display the HSV values
        print(f"Average HSV: H={h_avg}, S={s_avg}, V={v_avg}")
        
        # Visualize the color
        display_img = np.zeros((100, 100, 3), dtype=np.uint8)
        display_img[:, :] = cv2.cvtColor(np.uint8([[[h_avg, s_avg, v_avg]]]), cv2.COLOR_HSV2BGR)[0][0]
        cv2.putText(display_img, f"H:{h_avg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"S:{s_avg}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"V:{v_avg}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Average Color", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return (h_avg, s_avg, v_avg)
    
    return None


def generate_hsv_range(center, offset):
    """
    Generate one or two HSV bounds around a center HSV value, handling wrap-around in H.
    """
    h, s, v = center
    h_off, s_off, v_off = offset

    # initial, unclamped ranges
    lower = [h - h_off, max(s - s_off, 0), max(v - v_off, 0)]
    upper = [h + h_off, min(s + s_off, 255), min(v + v_off, 255)]

    # split into one or two ranges if H wraps
    ranges = []
    if lower[0] < 0:
        # wrap at low end: [0..upper] and [180+lower..179]
        ranges.append(
            ([0, lower[1], lower[2]],
             [upper[0], upper[1], upper[2]])
        )
        wrap_low = 180 + lower[0]
        ranges.append(
            ([wrap_low, lower[1], lower[2]],
             [179, upper[1], upper[2]])
        )
    elif upper[0] > 179:
        # wrap at high end: [0..upper-180] and [lower..179]
        wrap_high = upper[0] - 180
        ranges.append(
            ([0, lower[1], lower[2]],
             [wrap_high, upper[1], upper[2]])
        )
        ranges.append(
            ([lower[0], lower[1], lower[2]],
             [179, upper[1], upper[2]])
        )
    else:
        ranges.append((lower, upper))

    # Create a blank image to display the colors
    display_img = np.zeros((100, 300, 3), dtype=np.uint8)

    # Convert HSV values to BGR for display
    center_bgr = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2BGR)[0][0]
    lower_bgr = cv2.cvtColor(np.uint8([[ranges[0][0]]]), cv2.COLOR_HSV2BGR)[0][0]
    upper_bgr = cv2.cvtColor(np.uint8([[ranges[0][1]]]), cv2.COLOR_HSV2BGR)[0][0]

    # Draw rectangles for lower, center, and upper HSV values
    display_img[:, :100] = lower_bgr
    display_img[:, 100:200] = center_bgr
    display_img[:, 200:] = upper_bgr

    # Add text labels
    cv2.putText(display_img, 'Lower', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_img, 'Center', (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_img, 'Upper', (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image
    cv2.imshow('HSV Ranges', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ranges


# load or initialize color_ranges
script_dir = os.path.dirname(__file__)
ranges_path = os.path.join(script_dir, 'color_ranges.json')
if os.path.exists(ranges_path):
    with open(ranges_path, 'r') as f:
        color_ranges = json.load(f)
    print(f"Loaded color ranges from {ranges_path}")
else:
    color_ranges = {}
    for color in ['red', 'blue', 'yellow', 'green']:
        file = input(f"Select file for {color} box: ")
        selected = select_color_region(file)
        if selected is not None:
            h, s, v = selected
            ranges = generate_hsv_range((h, s, v), (10, 40, 40))
            color_ranges[color] = ranges
            for lower, upper in ranges:
                print(f"{color.upper()} HSV Range: Lower {lower}, Upper {upper}")
        else:
            print(f"Failed to select region for {color} box.")
    with open(ranges_path, 'w') as f:
        json.dump(color_ranges, f)


def detect_boxes(image_path, display=False):
    """
    Detects red, blue, yellow, and green boxes in the image and returns their order from left to right.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image at {image_path}")
        return []

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detections = []

    for color, ranges in color_ranges.items():
        mask = None
        # Build mask for this color
        for lower, upper in ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # show the mask for debugging
        if display:
            cv2.imshow(f'Mask for {color}', mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            # filter out boxes where width > 0.67 * height
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 2 * h:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            detections.append((color, cx, cy, cnt, area))

            if display:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.drawContours(img, [cnt], -1, (255, 0, 0), 3)
                
    # keep only the two biggest boxes by area
    if len(detections) > 2:
        detections.sort(key=lambda t: t[4], reverse=True)
        detections = detections[:2]

    # now sort those two by x (left→right)
    detections.sort(key=lambda t: t[1])
    order = [d[0] for d in detections]

    if display:
        cv2.imshow('Detected Boxes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return order


def main():
    parser = argparse.ArgumentParser(description='Detect colored boxes and list their order')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--display', action='store_true', help='Display detected boxes on image')
    args = parser.parse_args()

    order = detect_boxes(args.image, args.display)
    print('Detected color order:', order)


if __name__ == '__main__':
    main()
