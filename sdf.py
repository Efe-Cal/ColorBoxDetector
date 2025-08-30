import cv2
import numpy as np
import glob
import os

# Paths
objects = ["pics/green.png","pics/red.png","pics/yellow.png","pics/blue.png",]  # folder with object images
scene_path = "newpics/deneme2.png"     # the scene where objects should be detected

# Load scene image
scene_img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
scene_img_color = cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)

# Check if scene image loaded properly
if scene_img is None:
    print(f"Error: Could not load scene image from {scene_path}")
    exit()

print(f"Scene image loaded: {scene_img.shape}")

# Initialize SIFT with more features
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
kp_scene, des_scene = sift.detectAndCompute(scene_img, None)

print(f"Scene keypoints detected: {len(kp_scene)}")
if des_scene is None:
    print("Error: No descriptors found in scene image")
    exit()

# Matcher - using FLANN for better performance
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


for obj_path in objects:  
    print(f"\nProcessing: {os.path.basename(obj_path)}")
    obj_img = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
    if obj_img is None:
        print(f"  Error loading {obj_path}")
        continue

    print(f"  Object image shape: {obj_img.shape}")
    kp_obj, des_obj = sift.detectAndCompute(obj_img, None)
    if des_obj is None or len(kp_obj) < 4:
        print(f"  Not enough keypoints found: {len(kp_obj) if kp_obj else 0}")
        continue
    
    print(f"  Object keypoints: {len(kp_obj)}")

    # Use FLANN matcher for better results
    matches = flann.knnMatch(des_obj, des_scene, k=2)

    # Lowe's ratio test - more lenient threshold
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:  # More lenient threshold
                good.append(m)

    print(f"  Good matches found: {len(good)}")

    MIN_MATCH_COUNT = 4  # Reduced minimum match count
    if len(good) >= MIN_MATCH_COUNT:
        # Homography with more robust parameters
        src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.995)

        if H is not None:
            h, w = obj_img.shape
            pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, H)

            # Check if the detection is reasonable (not too distorted)
            area = cv2.contourArea(dst)
            original_area = w * h
            if area > original_area * 0.1 and area < original_area * 10:  # Reasonable size bounds
                # Draw detection box
                scene_img_color = cv2.polylines(scene_img_color, [np.int32(dst)], True, (0,255,0), 3)
                
                # Add label
                center = np.mean(dst, axis=0)[0].astype(int)
                cv2.putText(scene_img_color, os.path.basename(obj_path), 
                           tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                print(f"  ✓ DETECTED: {os.path.basename(obj_path)} (area ratio: {area/original_area:.2f})")
            else:
                print(f"  Detection rejected due to unreasonable size (area ratio: {area/original_area:.2f})")
        else:
            print(f"  Could not compute homography")
    else:
        print(f"  Not enough matches for {os.path.basename(obj_path)} ({len(good)}/{MIN_MATCH_COUNT})")

# Show final scene with detections
print(f"\nShowing results...")
cv2.imshow("Detected Objects", scene_img_color)

# Also show scene with keypoints for debugging
scene_with_kp = cv2.drawKeypoints(scene_img, kp_scene, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Scene Keypoints", scene_with_kp)

print("Press any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()