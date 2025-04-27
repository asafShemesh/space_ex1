import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()  # מדוד זמן ריצה

# Read the image
image = cv2.imread(r"C:\space_ex1\IMG_3046.jpg", cv2.IMREAD_GRAYSCALE)

# Apply a slight Gaussian Blur to reduce noise
image_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Set up the SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Thresholds
params.minThreshold = 100
params.maxThreshold = 255

# Filter by Area
params.filterByArea = True
params.minArea = 5
params.maxArea = 3000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.25

# Filter by Color
params.filterByColor = True
params.blobColor = 255

# Disable other filters
params.filterByConvexity = False
params.filterByInertia = False

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image_blur)

# Create a copy of the original image for drawing
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Open a file to save the star information
with open(r"C:\space_ex1\3046.txt", 'w') as f:
    f.write("Star_Number,X,Y,Radius,Brightness\n")  # header

    star_counter = 1

    # Process each detected star
    for keypoint in keypoints:
        cX = int(keypoint.pt[0])
        cY = int(keypoint.pt[1])
        radius = int(keypoint.size / 2)

        # Faster brightness calculation by cropping small region
        r = int(max(5, radius))
        x1 = max(cX - r, 0)
        x2 = min(cX + r, image.shape[1] - 1)
        y1 = max(cY - r, 0)
        y2 = min(cY + r, image.shape[0] - 1)
        star_region = image[y1:y2, x1:x2]
        brightness = np.mean(star_region)

        # Optional: filter by brightness
        if brightness < 80:
            continue

        # Draw circle at center position
        cv2.circle(result, (cX, cY), radius, (0, 255, 0), 2)

        # Write star information to file
        f.write(f"{star_counter},{cX},{cY},{radius},{brightness:.2f}\n")
        star_counter += 1

# Display result
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f'Detected {star_counter-1} Stars (fast detection)')
plt.show()

# Print runtime
end_time = time.time()
print(f"\nFinished in {end_time - start_time:.2f} seconds.")
