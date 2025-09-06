# Code for generating color ranges based on user-selected regions in an image

import cv2
import numpy as np
import os

script_dir = os.path.dirname(__file__)
color_ranges_path = os.path.join(script_dir, 'color_ranges.json')
HSV_OFFSET = (10, 55, 55) # Offset for HSV range generation

def select_color_region_hsv_average(img, title="Select Color Region"):
    """
    Opens an image and allows the user to select a rectangular region.
    Returns the average HSV value of the selected area.

    Args:
        image (np.ndarray): Image array (BGR)

    Returns:
        tuple: (h, s, v) average values of the selected region

    Usage:
        Press and drag to select a region
        Press 'c' to confirm selection
        Press 'r' to reset selection
        Press 'q' to quit without selection
    """
    # Use select_roi_from_image to get ROI coordinates
    # Save image temporarily if needed, but here we assume select_roi_from_image can take an image array
    if img is None:
        print(f"Error: Image is None")
        return None
    
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print("Instructions: Drag to select a region, press 'c' to confirm, 'r' to reset, 'q' to quit")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    # Create window and set mouse callback
    window_name = title
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
        cv2.imshow(window_name, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
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
    print(f"Selected region from {rect_start} to {rect_end}")
    
    # Check if we have valid coordinates
    if rect_start is None or rect_end is None:
        print("Error: No valid selection made. rect_start or rect_end is None")
        return None
    
    # If we have a valid selection
    if rect_start and rect_end:
        # Ensure correct order of coordinates (top-left, bottom-right)
        x1, y1 = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
        x2, y2 = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])
        
        print(f"Selection coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Check if the selection has valid dimensions
        if x2 <= x1 or y2 <= y1:
            print("Error: Invalid selection dimensions")
            return None
        
        # Extract the region
        region = img[y1:y2, x1:x2]
        print(f"Region shape: {region.shape}")
        
        if region.size == 0:
            print("Error: Selected region is empty")
            return None
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)

        # Compute circular mean for H to handle wrap‐around at 0/180
        h_vals = hsv_region[:, :, 0].astype(np.float32)
        # OpenCV H: 0–179 maps to 0–360°
        angles = np.deg2rad(h_vals * 2.0)
        mean_x = np.mean(np.cos(angles))
        mean_y = np.mean(np.sin(angles))
        mean_angle = np.arctan2(mean_y, mean_x)
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        # back to OpenCV H range
        h_avg = int(np.round(np.rad2deg(mean_angle) / 2.0)) % 180

        # S and V are linear
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
        print(f"Returning HSV values: ({h_avg}, {s_avg}, {v_avg})")
        return (h_avg, s_avg, v_avg)
    
    print("Error: No valid selection was made")
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

color_codes = {
    "red":"\033[31m",
    "blue":"\033[34m",
    "yellow":"\033[93m",
    "green":"\033[92m",
    "reset":"\033[0m"
}

def generate_color_ranges(img,title="Select Color Region"):
    color = select_color_region_hsv_average(img,title=title)
    print(f"Color selection result: {color}")
    
    if color is None:
        print("Error: Color selection failed. Cannot generate color ranges.")
        return None
    
    color_range = generate_hsv_range(color, HSV_OFFSET)
    
    # blue_loc = input(f"Select the location of the {color_codes['blue']}blue{color_codes['reset']} box (left/right): ").strip().lower()
    # blue = select_color_region_hsv_average(left_box_image if blue_loc == 'l' else right_box_image, "the blue box area")
    # blue_range = generate_hsv_range(blue, HSV_OFFSET)
    
    # green_loc = input(f"Select the location of the {color_codes['green']}green{color_codes['reset']} box (left/right): ").strip().lower()
    # green = select_color_region_hsv_average(left_box_image if green_loc == 'l' else right_box_image, "the green box area")
    # green_range = generate_hsv_range(green, HSV_OFFSET)
    
    # yellow_loc = input(f"Select the location of the {color_codes['yellow']}yellow{color_codes['reset']} box (left/right): ").strip().lower()
    # yellow = select_color_region_hsv_average(left_box_image if yellow_loc == 'l' else right_box_image, "the yellow box area")
    # yellow_range = generate_hsv_range(yellow, HSV_OFFSET)
    
    print(color_range)
    return color_range
    
if __name__ == "__main__":
    # generate_color_ranges(np.zeros((200, 200, 3), dtype=np.uint8),np.zeros((400, 200, 3), dtype=np.uint8))
    image = cv2.imread(input("Enter img path: "))  # Replace with your image path
    # select_color_region_old("image.jpg")  # You can also use the new function if you want to test it
    x= generate_color_ranges(image)
    print(x)