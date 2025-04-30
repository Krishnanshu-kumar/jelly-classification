import cv2
import numpy as np


def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized = cv2.resize(gray, (128, 128))

    # Normalize image
    normalized = resized / 255.0

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Apply Histogram Equalization
    equalized = cv2.equalizeHist(resized)

    # Apply Canny Edge Detection
    edges = cv2.Canny(resized, 100, 200)

    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Morphological Operations
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(resized, kernel, iterations=1)
    dilated = cv2.dilate(resized, kernel, iterations=1)

    # Stack images for a single window display
    images = [resized, blurred, equalized, edges, adaptive_thresh, eroded, dilated]
    labels = ["Resized", "Blurred", "Equalized", "Edges", "Threshold", "Eroded", "Dilated"]

    stacked_image = np.hstack(
        [cv2.putText(img.copy(), lbl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA) for img, lbl in
         zip(images, labels)])

    # Resize window to make it bigger
    cv2.namedWindow("Image Processing Steps", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Processing Steps", 1000, 400)

    cv2.imshow("Image Processing Steps", stacked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = "01.jpg"  # Replace with the path to your image
preprocess_image(image_path)
