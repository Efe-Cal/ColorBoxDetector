import json
import cv2
from select_mid_point import select_vertical_point
from selectROI import select_roi_from_image
from colorbox_color_generator import generate_color_ranges

first_image_path = input("Enter the path to the image file 1: ").strip()
second_image_path = input("Enter the path to the image file 2: ").strip()
if not first_image_path or not second_image_path:
    raise ValueError("Image path cannot be empty.")
first_image = cv2.imread(first_image_path)
second_image = cv2.imread(second_image_path)

def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

# All configurations will be saved in this file
config_file = 'config.json'

# Select areas
big_box_crop = select_roi_from_image(first_image, "Select the big box area")
left_box_crop = select_roi_from_image(second_image, "Select the left box area")
right_box_crop = select_roi_from_image(first_image, "Select the right box area")

# Gennerate images of regions
left_box_image = crop_image(second_image, *left_box_crop)
right_box_image = crop_image(first_image, *right_box_crop)

# print("Shapes",left_box_image.shape, right_box_image.shape)

cv2.imshow("Left Box Image", left_box_image)
cv2.imshow("Right Box Image", right_box_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get mid points for left and right box areas
left_box_mid_x = select_vertical_point(left_box_image)
right_box_mid_x = select_vertical_point(right_box_image)

# Get color ranges for the boxes
color_ranges = generate_color_ranges(left_box_image, right_box_image)

config_data = {
    "big_box_crop": big_box_crop,
    "left_box_crop": left_box_crop,
    "right_box_crop": right_box_crop,
    "left_box_mid_x": left_box_mid_x,
    "right_box_mid_x": right_box_mid_x,
    "color_ranges": color_ranges
}

with open(config_file, 'w') as f:
    json.dump(config_data, f)

print(f"Configuration saved to {config_file}")
print(color_ranges)