import cv2
import numpy as np
import matplotlib.pyplot as plt
def transformation1(image, T):
    return np.where(image < T, 0, 128).astype(np.uint8)

# Function 2: Linear to 128, then constant
def transformation2(image, T):
    return np.where(image < T, (128 / T) * image, 128).astype(np.uint8)

# Function 3: Zero until T, then linear to 255
def transformation3(image, T):
    return np.where(image < T, 0, (255 / (255 - T)) * (image - T)).astype(np.uint8)

# Load grayscale image
img1 = cv2.imread(r'E:\dipimage\tulip1.jpg', cv2.IMREAD_GRAYSCALE)

#if img is None:
  #  raise ValueError("Image not found. Check the file path.")

# Apply custom threshold for three values
T1, T2, T3 = 80, 150, 180
binary1 = transformation1(img1, T1)
binary2 = transformation1(img1, T2)
binary3 = transformation1(img1, T3)
img2 = cv2.imread(r'E:\dipimage\sapla.jpg', cv2.IMREAD_GRAYSCALE)
# Apply multi-thresholding
multi_thresh = transformation2(img2, T1)
multi_thresh2 = transformation2(img2, T2)
multi_thresh3 =transformation2(img2, T3)
img3 = cv2.imread(r'E:\DIP\rose2.jpg', cv2.IMREAD_GRAYSCALE)
t1, t2, t3 = 30, 50, 80
sigmoid_thresholded = transformation3(img3, t1)
sigmoid_thresholded2 = transformation3(img3, t2)
sigmoid_thresholded3 = transformation3(img3, t3)

# Plot all results
plt.figure(figsize=(20, 20))

plt.subplot(3, 7, 1)
plt.imshow(img1, cmap='gray')
plt.title("Original Image")
plt.axis('off')


plt.subplot(3, 7, 2)
plt.imshow(binary1, cmap='gray')
plt.title(f"Thresholded (T={T1})")
plt.axis('off')
plt.subplot(3, 7, 3)
plt.hist(binary1.ravel(), bins=256, range=(0, 255), color='black')
plt.title("Histogram")
plt.subplot(3, 7, 4)
plt.imshow(binary2, cmap='gray')
plt.title(f"Thresholded (T={T2})")
plt.axis('off')
plt.subplot(3, 7, 5)
plt.hist(binary2.ravel(), bins=256, range=(0, 255), color='black')
plt.subplot(3, 7, 6)
plt.imshow(binary3, cmap='gray')
plt.title(f"Thresholded (T={T3})")
plt.axis('off')
plt.subplot(3, 7, 7)
plt.hist(binary3.ravel(), bins=256, range=(0, 255), color='black')
plt.subplot(3, 7, 8)
plt.imshow(img2, cmap='gray')
plt.title("Image2")
plt.axis('off')

plt.subplot(3, 7, 9)
plt.imshow(multi_thresh, cmap='gray')
plt.title('Transformation1')
plt.axis('off')
plt.subplot(3, 7, 10)
plt.hist(multi_thresh.ravel(), bins=256, range=(0, 255), color='black')
plt.subplot(3, 7, 11)
plt.imshow(multi_thresh2, cmap='gray')
plt.title('Transformation2')
plt.axis('off')
plt.subplot(3, 7, 12)
plt.hist(multi_thresh2.ravel(), bins=256, range=(0, 255), color='black')
plt.subplot(3, 7, 13)
plt.imshow(multi_thresh3, cmap='gray')
plt.title('Transformation3')
plt.axis('off')
plt.subplot(3, 7, 14)
plt.hist(multi_thresh3.ravel(), bins=256, range=(0, 255), color='black')
plt.subplot(3, 7, 15)
plt.imshow(img3, cmap='gray')
plt.title('Image3')
plt.axis('off')
plt.subplot(3, 7, 16)
plt.imshow(sigmoid_thresholded, cmap='gray')
plt.title('Transformation1')
plt.axis('off')
plt.subplot(3, 7, 17)
plt.hist(sigmoid_thresholded.ravel(), bins=256, range=(0, 255), color='black')
plt.subplot(3, 7, 18)
plt.imshow(sigmoid_thresholded2, cmap='gray')
plt.title('Transformation2')
plt.axis('off')
plt.subplot(3, 7, 19)
plt.hist(sigmoid_thresholded2.ravel(), bins=256, range=(0, 255), color='black')
plt.subplot(3, 7, 20)
plt.imshow(sigmoid_thresholded3, cmap='gray')
plt.title('Tranformation3')
plt.axis('off')
plt.subplot(3, 7, 21)
plt.hist(sigmoid_thresholded3.ravel(), bins=256, range=(0, 255), color='black')
plt.tight_layout()
plt.show()
