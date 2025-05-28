import cv2

def select_roi_from_image(image,title="Select ROI"):
    print("Selecting ROI from image...")
    if isinstance(image,str): image = cv2.imread(image)
    if image is None:
        print("Error: Image not found or unable to load.")
        return None
    # Resize for display if image is too large
    max_display_size = 800  # Change this value as needed
    height, width = image.shape[:2]
    scale = min(max_display_size / width, max_display_size / height, 1.0)
    display_image = cv2.resize(image, (int(width * scale), int(height * scale)))
    # Let user select ROI from the resized image
    cv2.imshow(title, display_image)
    roi = cv2.selectROI(title, display_image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    # Scale ROI coordinates back to original size
    x, y, w, h = [int(coord / scale) for coord in roi]
    if x==0 and y==0 and w==0 and h==0:
        select_roi_from_image(image, title)  # Retry if no ROI was selected
    print(f"Selected ROI: {x}, {y}, {w}, {h}")
    # Optionally, extract and show the selected ROI from the original image
    if w > 0 and h > 0:
        selected_region = image[y:y+h, x:x+w]
        cv2.imshow(title, selected_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return (x, y, w, h)

if __name__ == "__main__":
    image_path = input("Enter image path:")  # Replace with your image path
    x = select_roi_from_image(image_path)
    print(f"Selected ROI: {x}")
