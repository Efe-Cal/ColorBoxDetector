import glob
import subprocess
import sys
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
img_path = './image.png'

def crop_image(img,x1,y1,x2,y2):
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img
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

def detect_and_extract_contours(img_path):
    contour_pos={}
    for ch in channels:
        img = cv2.imread(img_path)

        img = crop_image(img,1480,700,1676,835)
        # cv2.imshow('Cropped Image', img)

        # Resize while keeping aspect ratio
        h, w = img.shape[:2]
        max_w, max_h = 600, 600
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        result = cv2.resize(img, (new_w, new_h))
        
        result = isolate_and_subtract_channel(result, ch)

        # Extract the relevant channel as a single-channel image for contour detection
        channel_idx = {"r": 2, "g": 1, "b": 0}[ch]
        single_channel = result[:, :, channel_idx]
        # cv2.imshow(f'{channel_names[ch]} Channel', single_channel)

        if ch=="r":
            contours, mask = find_yellow_contours(img)
            # cv2.imshow('Yellow Mask', mask)
            cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    contour_pos["y"] = (cx, cy)
                cv2.drawContours(img, [largest_contour], -1, (0, 255, 255), 2)
                # cv2.imshow('Largest Yellow Contour', img)
                return "y"
        # Noise reduction: Apply Gaussian blur before thresholding
        # single_channel_blur = cv2.GaussianBlur(single_channel, (5, 5), 0)

        # cv2.imshow(f'Blurred {channel_names[ch]} Channel', single_channel_blur)
        # cv2.waitKey(0)
        # Thresholding
        _, single_channel_thresh = cv2.threshold(single_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        morph_open = cv2.morphologyEx(single_channel_thresh, cv2.MORPH_OPEN, np.ones(MORPH_KERNEL, np.uint8))
        closed = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, np.ones(MORPH_KERNEL, np.uint8))

        # Convert closed image to BGR for colored drawing
        result_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(result_bgr, contours, -1, (255, 0, 0), 1)  # Draw contours in blue
        # if not contours:
        #     raise ValueError(f"No contours found inqqqqq the {channel_names[ch]} channel.")

        # Debug: Print contour areas
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(f"Contour area in {channel_names[ch]} channel: {area}")

        # Filter contours based on area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 40000]

        # Filter contours based on aspect ratio
        ratio = 1.5
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] < ratio]
        # Find contour with the highest mean intensity in the relevant channel
        brightest_contour = max(contours, key=lambda cnt: cv2.mean(single_channel_thresh, mask=cv2.drawContours(np.zeros_like(single_channel_thresh), [cnt], -1, 255, -1))[0]) if contours else None

        cv2.drawContours(result_bgr, [brightest_contour], -1, (0, 255, 255), 2)  # Highlight brightest contour in yellow
        if brightest_contour is not None:
            print(ch)
            return ch
        # if brightest_contour is not None:
        #     M = cv2.moments(brightest_contour)
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])

        #     contour_pos[ch] = (cx, cy)

        # cv2.imshow(f'Isolated {channel_names[ch]} Channel', result_bgr)
    print("blue")
    return "b"

def image_job():
    subprocess.run(["rpicam-still", "--output", img_path, "--timeout", "200", "--width", "1920", "--height", "1080", "--rotation", "180"])
    return detect_and_extract_contours(img_path)