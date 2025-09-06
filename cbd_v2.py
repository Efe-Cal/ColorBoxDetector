import cv2
import numpy as np

MORPH_KERNEL = (5, 5)  # Kernel size for morphological operations

def isolate_and_subtract_channel(img, channel='r'):
    
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


# Example usage:
channels = ['r', 'g', 'b']
channel_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
img_path = r'C:\Users\efeca\Desktop\new_image.png'

contour_pos={}
for ch in channels:
    img = cv2.imread(img_path)

    result = isolate_and_subtract_channel(img, ch)

    # Resize while keeping aspect ratio
    h, w = result.shape[:2]
    max_w, max_h = 600, 600
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    result = cv2.resize(result, (new_w, new_h))
    

    # Extract the relevant channel as a single-channel image for contour detection
    channel_idx = {"r": 2, "g": 1, "b": 0}[ch]
    single_channel = result[:, :, channel_idx]

    # Noise reduction: Apply Gaussian blur before thresholding
    single_channel_blur = cv2.GaussianBlur(single_channel, (5, 5), 0)

    # Thresholding
    _, single_channel_thresh = cv2.threshold(single_channel_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    morph_open = cv2.morphologyEx(single_channel_thresh, cv2.MORPH_OPEN, np.ones(MORPH_KERNEL, np.uint8))
    closed = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, np.ones(MORPH_KERNEL, np.uint8))

    # Convert closed image to BGR for colored drawing
    result_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(result_bgr, contours, -1, (255, 0, 0), 1)  # Draw contours in blue
    if not contours:
        raise ValueError(f"No contours found in the {channel_names[ch]} channel.")

    # Debug: Print contour areas
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f"Contour area in {channel_names[ch]} channel: {area}")

    # Filter contours based on area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]

    # Filter contours based on aspect ratio
    ratio = 1.5
    contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] < ratio]
    # Find contour with the highest mean intensity in the relevant channel
    brightest_contour = max(contours, key=lambda cnt: cv2.mean(single_channel_thresh, mask=cv2.drawContours(np.zeros_like(single_channel_thresh), [cnt], -1, 255, -1))[0])

    cv2.drawContours(result_bgr, [brightest_contour], -1, (0, 255, 255), 2)  # Highlight brightest contour in yellow

    M = cv2.moments(brightest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    contour_pos[ch] = (cx, cy)

    cv2.imshow(f'Isolated {channel_names[ch]} Channel', result_bgr)

# Find the two contours with the closest cy values
channels_with_cy = [(ch, pos[1]) for ch, pos in contour_pos.items()]
min_diff = float('inf')
closest_pair = None

for i in range(len(channels_with_cy)):
    for j in range(i + 1, len(channels_with_cy)):
        diff = abs(channels_with_cy[i][1] - channels_with_cy[j][1])
        if diff < min_diff:
            min_diff = diff
            closest_pair = (channels_with_cy[i][0], channels_with_cy[j][0])

print(f"The two channels with the closest cy values are: {closest_pair}")

remaining_contour = [ch for ch in channels if ch not in closest_pair][0]

point_A_color = closest_pair[0]
point_B_color = closest_pair[1]
point_C_color = remaining_contour

A = (contour_pos[point_A_color][0], contour_pos[point_A_color][1])
B = (contour_pos[point_B_color][0], contour_pos[point_B_color][1])
C = (contour_pos[point_C_color][0], contour_pos[point_C_color][1])

D1 = (B[0] + C[0] - A[0], B[1] + C[1] - A[1])  # Opposite A 
D2 = (A[0] + C[0] - B[0], A[1] + C[1] - B[1])  # Opposite B

# show original resized image with D point
img = cv2.imread(img_path)
result = cv2.resize(img, (new_w, new_h))

# Select the area within 50 pixels radius of point D1
mask_D1 = np.zeros(result.shape[:2], dtype=np.uint8)
cv2.circle(mask_D1, D1, 50, 255, -1)
selected_area_D1 = cv2.bitwise_and(result, result, mask=mask_D1)

# Select the area within 50 pixels radius of point D2
mask_D2 = np.zeros(result.shape[:2], dtype=np.uint8)
cv2.circle(mask_D2, D2, 50, 255, -1)
selected_area_D2 = cv2.bitwise_and(result, result, mask=mask_D2)

# Find yellow contours in the selected areas around D1 and D2
def find_yellow_contours(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define yellow color range in HSV
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Morphological operations to clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area and aspect ratio
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]
    ratio = 1.5
    contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] < ratio]
    if not contours:
        return [], mask
        
    return contours, mask

yellow_contours_D1, yellow_mask_D1 = find_yellow_contours(selected_area_D1)
yellow_contours_D2, yellow_mask_D2 = find_yellow_contours(selected_area_D2)

if yellow_contours_D1 and len(yellow_contours_D1) > 0:
    selected_area_D1_contours = selected_area_D1.copy()
    cv2.drawContours(selected_area_D1_contours, yellow_contours_D1, -1, (255, 0, 0), 2)
    cv2.imshow('Yellow Contours around D1', selected_area_D1_contours)
    print("Contours found around D1")
    M = cv2.moments(yellow_contours_D1[0])
    y_cx = int(M['m10'] / M['m00'])
    y_cy = int(M['m01'] / M['m00'])

if yellow_contours_D2 and len(yellow_contours_D2) > 0:
    selected_area_D2_contours = selected_area_D2.copy()
    cv2.drawContours(selected_area_D2_contours, yellow_contours_D2, -1, (255, 0, 0), 2)
    cv2.imshow('Yellow Contours around D2', selected_area_D2_contours)
    print("Contours found around D2")
    M = cv2.moments(yellow_contours_D2[0])
    y_cx = int(M['m10'] / M['m00'])
    y_cy = int(M['m01'] / M['m00'])
    
cv2.circle(result, (y_cx, y_cy), 5, (0, 255, 255), -1)
cv2.imshow('Circle at yellow contour center', result)


# Identify which point is which corner of the parallelogram
points = {point_A_color: A, point_B_color: B, point_C_color: C, 'Y': (y_cx, y_cy)}

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

# corner_names = {
#     top_left[0]: "top left",
#     top_right[0]: "top right",
#     bottom_left[0]: "bottom left",
#     bottom_right[0]: "bottom right"
# }

# for name, pt in points.items():
#     print(f"{name} is {corner_names[name]} corner: {pt}")

r = ",".join([c[0].lower() for c in [top_left, top_right, bottom_right, bottom_left]])
print(r)
cv2.waitKey(0)
cv2.destroyAllWindows()