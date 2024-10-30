import cv2
import numpy as np
import os

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Directory containing images
directory = '/home/zy/Desktop/collected_images/'
image_filenames = ['test1.png', 'test2.png', 'test3.png', 'test4.png']
image_paths = [os.path.join(directory, name) for name in image_filenames]

# Dictionary to hold images, keypoints and descriptors
image_features = {}

# Loop over the image paths
for image_path in image_paths:
    # Load image
    image = cv2.imread(image_path)
    # Convert it to grayscale (if it is not already in that format)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect features and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # Store the keypoints and descriptors
    image_features[image_path] = (keypoints, descriptors)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Function to match and draw matches between two images
def match_and_draw(image1_path, image2_path):
    keypoints1, descriptors1 = image_features[image1_path]
    keypoints2, descriptors2 = image_features[image2_path]
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=2)
    return matched_image

# Match features and draw matches for each pair of consecutive images
for i in range(len(image_paths) - 1):
    matched_image = match_and_draw(image_paths[i], image_paths[i + 1])
    cv2.imshow(f'Matches {i+1}', matched_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

