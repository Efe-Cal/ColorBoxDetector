import cv2

# Load the image
image_path = input("Enter image path:")  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Resize for display if image is too large
    max_display_size = 800  # Change this value as needed
    height, width = image.shape[:2]
    scale = min(max_display_size / width, max_display_size / height, 1.0)

    display_image = cv2.resize(image, (int(width * scale), int(height * scale)))

    # Let user select ROI from the resized image
    cv2.imshow("Select ROI", display_image)
    roi = cv2.selectROI("Select ROI", display_image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # Scale ROI coordinates back to original size
    x, y, w, h = [int(coord / scale) for coord in roi]
    print(f"{x}, {y}, {w}, {h}")

    # Optionally, extract and show the selected ROI from the original image
    if w > 0 and h > 0:
        selected_region = image[y:y+h, x:x+w]
        cv2.imshow("Selected ROI", selected_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
