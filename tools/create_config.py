# Tool to create configuration file for color box detection

import json
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

def get_big_box_crop_area(image):
    big_box_crop = select_roi_from_image(image, "Select the big box area")
    config_data['big_box_crop'] = big_box_crop
    big_box_area_image = crop_image(image, *big_box_crop)
    return big_box_area_image

def get_single_color_range(img, color_name):
    image_path = input(f"Enter the path to the image file {color_name}: ").strip()
    image = cv2.imread(image_path)    

    big_box_area_image = crop_image(image, *config_data['big_box_crop'])

    color_ranges = generate_color_ranges(big_box_area_image)
    return color_ranges

bb_img = get_big_box_crop_area(cv2.imread(input("Enter the path to the image file for big box crop: ").strip()))
color_ranges = {}
for color in ['red', 'green', 'blue', 'yellow']:
    cr = get_single_color_range(bb_img, color)
    color_ranges[color] = cr
config_data['color_ranges'] = color_ranges
with open(config_file, 'w') as f:
    json.dump(config_data, f)

print(f"Configuration saved to {config_file}")
print(color_ranges)