import argparse
import cv2
import numpy as np
import os
import json

MORPHOLOGY_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
DIST_TRESH = 0.4  # Distance threshold for distance transform

def load_config():
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Please create it first.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def build_clean_mask(hsv: np.ndarray,
                     ranges: list[tuple[list[int],tuple[int]]],
                     kernel_size: tuple[int,int]=MORPHOLOGY_KERNEL_SIZE) -> np.ndarray:
    """Build and clean mask for a list of HSV ranges."""
    mask = None
    for lo, hi in ranges:
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def detect_boxes(image_path:str, display:bool) -> list[str]:
    """
    Detects red, blue, yellow, and green boxes in the image and returns their order from left to right.
    """
    if isinstance(image_path,str):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot load image at {image_path}")
            return []
    else: img=image_path
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detections = []

    for color, ranges in color_ranges.items():
        # Create mask for the color
        mask = build_clean_mask(hsv, ranges, kernel_size=MORPHOLOGY_KERNEL_SIZE)

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform,DIST_TRESH*dist_transform.max(),255,0)
        
        # show the mask for debugging
        if display:
            cv2.imshow('Distance Transform', dist_transform)
            cv2.resizeWindow('Distance Transform', 300, 300)
            cv2.moveWindow('Distance Transform', 40, 40)
            cv2.imshow('Sure foreground', sure_fg)
            cv2.resizeWindow('Sure foreground', 300, 300)
            cv2.moveWindow('Sure foreground', 300, 300)
            cv2.imshow(f'Mask for {color}', mask)
            cv2.resizeWindow(f'Mask for {color}', 300, 300)
            cv2.moveWindow(f'Mask for {color}', 500, 500)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Find contours
        sure_fg = sure_fg.astype(np.uint8) 
        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # if area < 300:
            #     print(f"Area too small: {area}")
            #     continue

            # filter out boxes where width > 0.67 * height
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 2 * h:
                print(f"Width too large: {w} > 2 * {h}")
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            detections.append((color, cx, cy, cnt, area))

            if display:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.drawContours(img, [cnt], -1, (255, 255, 255), 1)
                
    # keep only the two boxes at highest vertical position and most to the left
    if len(detections) > 2:
        # sort by cy (y coordinate), then by cx (x coordinate)
        detections.sort(key=lambda t: (t[2], t[1]))
        detections = detections[:2]

    # now sort those two by x (left→right)
    detections.sort(key=lambda t: t[1])
    order = [d[0] for d in detections]

    if display:
        cv2.imshow('Detected Boxes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return order

def crop_image(img, x:int, y:int, w:int, h:int) -> np.ndarray:
    return img[y:y+h, x:x+w]


def process_missing_boxes(left:list,right:list):
    all_colors={"red", "blue", "yellow", "green"}
    if len(right)+len(left)==4:
        pass
    elif len(right)+len(left)==3:
        if len(left)==1 and len(right)==2:
            left.append(list(all_colors - set(left + right)))
        elif len(left)==2 and len(right)==1:
            right.append(list(all_colors - set(left + right)))
    elif len(right)+len(left)==2:
        if len(left)==1 and len(right)==1:
            missing_colors = list(all_colors - set(left + right))
            right.append(missing_colors[0])
            left.append(missing_colors[1])
    elif len(right)+len(left)==1:
        missing_colors = list(all_colors - set(left + right))
        if len(left)==1:
            left.append(missing_colors[0])
            right.append(missing_colors[1:2])
        elif len(right)==1:
            right.append(missing_colors[0])
            left.append(missing_colors[1:2])

def get_boxes(img_path:str,display:bool=False) -> str:
    config = load_config()
    global color_ranges
    color_ranges = config["color_ranges"]
    crop_data = config["crop_data"]
    
    image = cv2.imread(img_path)
    
    big_box_image = crop_image(image, *crop_data["big_box_crop"])
    left_box_image = crop_image(image, *crop_data["left_box_crop"])
    right_box_image = crop_image(image, *crop_data["right_box_crop"])
    
    big_box_order = detect_boxes(big_box_image, display)
    if len(big_box_image)==0:
        big_box_order=["blue"]
    left_box_order = detect_boxes(left_box_image, display)
    right_box_order = detect_boxes(right_box_image, display)
    
    result_string = ";".join(big_box_order+left_box_order+right_box_order)
    return result_string

def main():
    parser = argparse.ArgumentParser(description='Detect colored boxes and list their order')
    parser.add_argument('image', nargs='?', help='Path to input image')
    parser.add_argument('--display', action='store_true', help='Display detected boxes on image')
    args = parser.parse_args()

    if not args.image:
        args.image = input("Enter path to input image: ").strip()

    order = get_boxes(args.image)
    print('Detected color order:', order)


if __name__ == '__main__':
    main()
