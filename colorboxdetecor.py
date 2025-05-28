import argparse
from typing import TypedDict
import cv2
import numpy as np
import os
import json

MORPHOLOGY_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
DIST_TRESH = 0.4  # Distance threshold for distance transform
EXTENSION_OFFSET = (10, 20, 20)  # Offset for extending color ranges

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


class Order(TypedDict):
    left: str
    right: str
    
def detect_boxes(img, mid_point,color_ranges, display:bool):
    """
    Detects red, blue, yellow, and green boxes in the image and returns their order from left to right.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detections = []
    order = Order(left="", right="")
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
        contour_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 70]

        area = sum(contour_areas)
        if area < 500:
            if not area == 0:
                print(f"Area too small: {area}   {color}")
            continue
        
        cnt = contours[np.argmax(contour_areas)]  # Get the largest contour
        
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    
        if cx < mid_point+5:
            order["left"] = color
        elif cx > mid_point-5:
            order["right"] = color

        # filter out boxes where width > 0.67 * height
        # if w > 2 * h:
        #     print(f"Width too large: {w} > 2 * {h}")
        #     continue

        detections.append(color)

        x, y, w, h = cv2.boundingRect(cnt)
        if display:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.drawContours(img, [cnt], -1, (255, 255, 255), 1)

    return [order,detections]

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
    color_ranges = config["color_ranges"]
    
    image = cv2.imread(img_path)
    
    big_box_image = crop_image(image, *config["big_box_crop"])
    left_box_image = crop_image(image, *config["left_box_crop"])
    right_box_image = crop_image(image, *config["right_box_crop"])

    # big_box_order = detect_boxes(big_box_image, display)
    # if len(big_box_image)==0:
    #     big_box_order=["blue"]
    left_box_order = detect_boxes(left_box_image, config["left_box_mid_x"], color_ranges,display)
    right_box_order = detect_boxes(right_box_image, config["right_box_mid_x"], color_ranges,display)
    
    if list(left_box_order[0].values()).count("") > 0:
        print("Extending color ranges for left box order")
        print("Left box order:", left_box_order)
        
        for key,value in color_ranges.items():
            color_ranges[key] = extend_color_range(value[0],EXTENSION_OFFSET)
        print("Color ranges after extension:", color_ranges)

        extended_left_box_order = detect_boxes(left_box_image, config["left_box_mid_x"], color_ranges,display)

        for k, v in extended_left_box_order[0].items():
            if left_box_order[0][k] == "" and v != "":
                left_box_order[0][k] = v

        left_box_order[1].extend(extended_left_box_order[1])
        left_box_order[1] = list(set(left_box_order[1]))  # remove duplicates
    
    if list(right_box_order[0].values()).count("") > 0:
        print("Extending color ranges for right box order")
        print("Right box order:", right_box_order)
        
        for key,value in color_ranges.items():
            color_ranges[key] = extend_color_range(value[0],EXTENSION_OFFSET)
        print("Color ranges after extension:", color_ranges)

        extended_right_box_order = detect_boxes(right_box_image, config["right_box_mid_x"], color_ranges,display)
        
        for k, v in extended_right_box_order[0].items():
            if right_box_order[0][k] == "" and v != "":
                right_box_order[0][k] = v
                
        right_box_order[1].extend(extended_right_box_order[1])
        right_box_order[1] = list(set(right_box_order[1]))
        
    print("Left box order:", left_box_order)
    print("Right box order:", right_box_order)
    return left_box_order, right_box_order

def main():
    parser = argparse.ArgumentParser(description='Detect colored boxes and list their order')
    parser.add_argument('image', nargs='?', help='Path to input image')
    parser.add_argument('--display', action='store_true', help='Display detected boxes on image')
    args = parser.parse_args()

    # if not args.image:
    #     args.image = input("Enter path to input image: ").strip()

    order = get_boxes(args.image,args.display)
    # TODO:
    # order = process_missing_boxes()
    # order = stringify_order(order)
    print('Detected color order:', order)


if __name__ == '__main__':
    main()
    # x = extend_color_range([
    #     [93, 157, 62],
    #     [123, 255, 172]
    #   ], EXTENSION_OFFSET)
    # print(x)