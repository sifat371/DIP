import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to slice bit planes (1st way: bitwise operations)
def bit_plane_slicing(image):
    bit_planes = []
    for i in range(8):
        # Extract each bit-plane and keep its grayscale intensity
        sliced_img = image & (1 << i)
        bit_planes.append(sliced_img)
    return bit_planes

# Function to display images in a grid
def display_imgset(img_set, color_set, title_set='', row=1, col=1):
    plt.figure(figsize=(20, 20))
    k = 1
    n = len(img_set)
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if k > n:
                break
            plt.subplot(row, col, k)
            plt.axis('off')
            img = img_set[k-1]
            if len(img.shape) == 3:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap=color_set[k-1])
            if title_set[k-1] != '':
                plt.title(title_set[k-1])
            k += 1
    plt.show()
    plt.close()

def main():
    # Load grayscale image
    img_path = r'E:\DIP\rose2.jpg'
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if gray_img is None:
        raise ValueError("Image not found.")

    # Slice into bit planes
    bit_planes = bit_plane_slicing(gray_img)

    # Combine some bit-planes for partial reconstruction
    cmd_planes123 = bit_planes[0] + bit_planes[1] + bit_planes[2]
    cmd_planes234 = bit_planes[1] + bit_planes[2] + bit_planes[3]
    cmd_planes456 = bit_planes[3] + bit_planes[4] + bit_planes[5]
    cmd_planes567 = bit_planes[4] + bit_planes[5] + bit_planes[6]
    cmd_planes678 = bit_planes[5] + bit_planes[6] + bit_planes[7]

    # Reconstruct the image
    reconstructed_img = sum(bit_planes)

    # Check lossless reconstruction
    loss = np.sum(gray_img - reconstructed_img)
    print(f"Loss value: {loss}")
    if loss == 0:
        print("Lossless reconstruction was done...")

    # Prepare display set
    img_set = [gray_img] + bit_planes + [
        cmd_planes123, cmd_planes234, cmd_planes456,
        cmd_planes567, cmd_planes678, reconstructed_img
    ]
    title_set = [
        'Original Image',
        'Plane_1', 'Plane_2', 'Plane_3', 'Plane_4',
        'Plane_5', 'Plane_6', 'Plane_7', 'Plane_8',
        'Plane_1+2+3', 'Plane_2+3+4', 'Plane_4+5+6',
        'Plane_5+6+7', 'Plane_6+7+8', 'All_Planes'
    ]
    color_set = ['gray'] * len(img_set)

    # Show images
    display_imgset(img_set, color_set, title_set, row=3, col=5)

if _name_ == "_main_":
    main()
