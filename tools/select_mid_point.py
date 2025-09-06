# Obsolete with the colorBoxDetector

import cv2

def select_vertical_point(image):
    """
    Open an image, let the user move the mouse to position a vertical line,
    and on left-click return the x-coordinate of the clicked pixel.
    """
    if image is None:
        raise FileNotFoundError(f"Image not found: {image}")
    clone = image.copy()
    x_coord = [None]
    clicked = [False]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            x_coord[0] = x
        elif event == cv2.EVENT_LBUTTONDOWN:
            x_coord[0] = x
            clicked[0] = True

    window_name = "Select Point"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = clone.copy()
        if x_coord[0] is not None:
            cv2.line(display, (x_coord[0], 0),
                     (x_coord[0], display.shape[0]),
                     (0, 255, 0), 1)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if clicked[0] or key == 27:  # ESC to cancel
            break

    cv2.destroyAllWindows()
    return x_coord[0]

if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ").strip()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")

    mid_x = select_vertical_point(image)
    print(f"Selected vertical mid-point x-coordinate: {mid_x}")
