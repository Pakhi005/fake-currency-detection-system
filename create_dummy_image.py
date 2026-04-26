import numpy as np
import cv2

# Create a blank white image
img = np.ones((500, 800, 3), dtype=np.uint8) * 255

# Draw a rectangle that has a ~2.5 aspect ratio (e.g., 600x240) to simulate a note
cv2.rectangle(img, (100, 100), (700, 340), (0, 120, 0), -1)

# Add some text
cv2.putText(img, "DUMMY CURRENCY NOTE", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Add a simulated security thread
cv2.line(img, (400, 100), (400, 340), (200, 200, 200), 5)

cv2.imwrite("your_test_image.jpg", img)
print("Dummy test image 'your_test_image.jpg' created successfully!")
