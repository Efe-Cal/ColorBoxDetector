import argparse
import glob
import subprocess
import cv2
import numpy as np
import os
import json

MORPHOLOGY_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
DIST_TRESH = 0.4  # Distance threshold for distance transform
EXTENSION_OFFSET = (10, 30, 30)  # Offset for extending color ranges

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

def extend_color_range(color_range: list, offset:tuple=(5,10,10)) -> list:
    """Extends a color range by an offset."""
    if isinstance(color_range[0][0],list):
        lo = extend_color_range(color_range[0], offset)[0]
        hi = extend_color_range(color_range[1], offset)[0]
    else:
        lo, hi = color_range
        lo = list(max(0, c - o) for c, o in zip(lo, offset))
        hi = list(min(255, c + o) for c, o in zip(hi, offset))
        
    return [[lo, hi]]

def detect_boxes(img,color_ranges, display:bool):
    """
    Detects red, blue, yellow, and green boxes in the image and returns their order from left to right.
    """
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
        contours, _ = cv2.findContours(dist_transform.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        contour_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 50]

        area = sum(contour_areas)
        print(area)
        if area < 500:
            if not area == 0:
                print(f"Area too small: {area}   {color}")
            continue
        
        cnt = contours[np.argmax(contour_areas)]  # Get the largest contour
        
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    

        detections.append((color,area,cx,cy))

        x, y, w, h = cv2.boundingRect(cnt)
        if display:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.drawContours(img, [cnt], -1, (255, 255, 255), 1)

    def closeness_to_center(detection):
        _,_,cx,cy = detection
        img_cx, img_cy = img.shape[1]//2, img.shape[0]//2
        return np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
    
    if len(detections) == 0:
        print("No boxes detected")
        return "b"
    elif len(detections) == 1:
        return detections[0][0][0]    
    elif len(detections) > 1:
        # Normalize area and closeness to center, then combine equally
        areas = np.array([d[1] for d in detections])
        centers = np.array([closeness_to_center(d) for d in detections])
        norm_areas = (areas - areas.min()) / (areas.ptp() if areas.ptp() > 0 else 1)
        norm_centers = (centers - centers.min()) / (centers.ptp() if centers.ptp() > 0 else 1)
        scores = norm_areas + (1 - norm_centers)  # larger area and closer to center preferred
        best_idx = np.argmax(scores)
        detection = detections[best_idx][0][0]
    
    return detection

def crop_image(img, x:int, y:int, w:int, h:int) -> np.ndarray:
    return img[y:y+h, x:x+w]
    

def get_box_color(img_path:str,display:bool=False) -> str:
    config = load_config()
    color_ranges = config["color_ranges"]
    
    image = cv2.imread(img_path)
    
    # big_box_image = crop_image(image1, *config["big_box_crop"])
    box_image = crop_image(image, *config["big_box_crop"])

    d = detect_boxes(box_image, color_ranges, display)
    print(d)
    return d


def main():
    parser = argparse.ArgumentParser(description='Detect colored boxes and list their order')
    parser.add_argument('image1', help='Path to the first input image')
    parser.add_argument('--display', action='store_true', help='Display detected boxes on image')
    args = parser.parse_args()
    get_box_color(args.image1, args.display)
    # order = get_boxes(args.image1, args.image2, args.display)
    # print('Detected color order:', order)

def image_job():
    # rpicam-still --output ./image.png --timeout 200 --width 1920 --height 1080 --rotation 180
    img_path = "./image.png"
    subprocess.run(["rpicam-still", "--output", img_path, "--timeout", "200", "--width", "1920", "--height", "1080", "--rotation", "180"])
    return get_box_color(img_path, display=False)

if __name__ == '__main__':
    for i in glob.glob("C:/Users/efeca/Desktop/np/*.png"):
        print(i)
        get_box_color(i)