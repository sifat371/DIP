import cv2
import numpy as np
import matplotlib.pyplot as plt

#================= Histogram Equalization Function ======================
def histogram_equalization_custom(img_gray):
    # Step 1: Calculate Histogram
    hist = np.zeros(256, dtype=int)
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            hist[img_gray[i, j]] += 1

    # Step 2: Calculate PDF
    pdf = hist / np.sum(hist)

    # Step 3: Calculate CDF
    cdf = np.cumsum(pdf)

    # Step 4: Mapping new intensity levels
    new_level = np.round(cdf * 255).astype(np.uint8)

    # Step 5: Apply mapping to image
    img_equalized = new_level[img_gray]

    return img_equalized, hist, pdf, cdf


#================= Display Function ================================
def display_images(original, equalized_custom, equalized_cv2, original_hist, equalized_hist_custom, equalized_hist_cv2):
    # Display original and equalized images along with their histograms in one window (3 rows, 2 columns)
    plt.figure(figsize=(12,15))

    # Row 1: Original Image and Original Histogram
    plt.subplot(3, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    plt.bar(range(256), original_hist)
    plt.title("Original Histogram")

    # Row 2: Equalized Image (Custom) and Custom Equalized Histogram
    plt.subplot(3, 2, 3)
    plt.imshow(equalized_custom, cmap='gray')
    plt.title("Equalized Image (Custom)")
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.bar(range(256), equalized_hist_custom)
    plt.title("Equalized Histogram (Custom)")

    # Row 3: Equalized Image (OpenCV) and OpenCV Equalized Histogram
    plt.subplot(3, 2, 5)
    plt.imshow(equalized_cv2, cmap='gray')
    plt.title("Equalized Image (OpenCV)")
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    plt.bar(range(256), equalized_hist_cv2)
    plt.title("Equalized Histogram (OpenCV)")

    plt.tight_layout()
    plt.show()


#================= Main Function ============================
def main():
    # Load grayscale image
    img_path = r"E:\dipimage\sapla.jpg"
    img_gray = cv2.imread(img_path, 0)

    # Apply custom histogram equalization
    img_equalized_custom, original_hist, pdf, cdf = histogram_equalization_custom(img_gray)

    # Apply OpenCV's built-in histogram equalization
    img_equalized_cv2 = cv2.equalizeHist(img_gray)

    # Get histograms for the equalized images
    equalized_hist_custom = np.histogram(img_equalized_custom, bins=256, range=(0, 255))[0]
    equalized_hist_cv2 = np.histogram(img_equalized_cv2, bins=256, range=(0, 255))[0]

    # Display results in one window with 3 rows and 2 columns
    display_images(img_gray, img_equalized_custom, img_equalized_cv2, original_hist, equalized_hist_custom, equalized_hist_cv2)

if _name_ == "_main_":
    main()
