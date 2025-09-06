# Config creation tool for parallelogram detection
 
import json
import cv2
from selectROI import select_roi_from_image
from colorbox_color_generator import generate_color_ranges

config_file = 'config_boxes.json'
def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]
config_data = {}

def get_crop_areas(image):
    boxes_crop = select_roi_from_image(image, "Select boxes area")
    config_data['boxes_crop'] = boxes_crop
    boxes_crop_image = crop_image(image, *boxes_crop)
    
    return boxes_crop_image

def get_single_color_range(color_name,img_path=None):
    image_path = input(f"Enter the path to the image file {color_name}: ").strip() if not img_path else img_path
    image = cv2.imread(image_path)    

    boxes_area_image = crop_image(image, *config_data['boxes_crop'])

    color_ranges = generate_color_ranges(boxes_area_image,title=f"Select {color_name} Color Region")
    return color_ranges

image_path = input("Enter the path to the image file for boxes crop: ").strip()

cropped_area = get_crop_areas(cv2.imread(image_path))
color_ranges = {}
for color in ['red', 'green', 'blue', 'yellow']:
    cr = get_single_color_range(color,image_path)
    color_ranges[color] = cr
config_data['color_ranges'] = color_ranges
with open(config_file, 'w') as f:
    json.dump(config_data, f)

print(f"Configuration saved to {config_file}")
print(color_ranges)