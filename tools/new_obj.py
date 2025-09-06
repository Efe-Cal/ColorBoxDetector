# Config creation tool for custom object

import json
import os
import cv2
from selectROI import select_roi_from_image
from colorbox_color_generator import generate_color_ranges

config_file = 'config.json'
def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]
config_data = {
    # "big_box_crop": big_box_crop,
    # "color_ranges": color_ranges
}

def get_big_box_crop_area(image,obj):
    big_box_crop = select_roi_from_image(image, "Select the area:::"+ obj)
    config_data[obj] = big_box_crop
    big_box_area_image = crop_image(image, * big_box_crop)
    return big_box_area_image

def get_single_color_range(color_name,obj):
    image_path = input(f"Enter the path to the image file {color_name}: ").strip()
    if image_path == '':
        return None
    image = cv2.imread(image_path)    

    big_box_area_image = crop_image(image, *config_data[obj])

    color_ranges = generate_color_ranges(big_box_area_image)
    return color_ranges

obj = input("Enter the object name: ").strip()

bb_img = get_big_box_crop_area(cv2.imread(input("Enter the path to the image file for obj crop: ").strip()),obj)
color_ranges = {}
for color in ['red', 'green', 'blue', 'yellow']:
    cr = get_single_color_range(color, obj)
    if cr is not None:
        color_ranges[color] = cr
if color_ranges:
    config_data['color_ranges'] = color_ranges
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        old_config = json.load(f)
        old_config.update(config_data)
    with open(config_file, 'w') as f:
        json.dump(old_config, f)
else:
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
print(f"Configuration saved to {config_file}")
print(color_ranges)