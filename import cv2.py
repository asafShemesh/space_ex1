import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/ben/Desktop/Space_Engineering/Assignment1/Photos_of_Stars/IMG_3046.jpg', cv2.IMREAD_GRAYSCALE)


_, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# Find contours of bright spots
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for drawing
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw circles around identified stars
for contour in contours:
    # Calculate center of contour
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Draw circle at center position
        cv2.circle(result, (cX, cY), 10, (0, 255, 0), 2)
        
        # You could also store these coordinates for further processing
        print(f"Star found at coordinates: ({cX}, {cY})")

# Display result
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Detected Stars')
plt.show()