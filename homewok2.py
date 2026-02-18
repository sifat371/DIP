import numpy as np
import matplotlib.pyplot as plt

# Image size
height, width = 100, 100

# Create color and grayscale images
black_image = np.zeros((height, width, 3), dtype=np.uint8)
gray1_image = np.ones((height, width, 3), dtype=np.uint8) * 100
gray2_image = np.ones((height, width, 3), dtype=np.uint8) * 150
gray3_image = np.ones((height, width, 3), dtype=np.uint8) * 240
white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
# Create red images with varying intensities
red1_image = np.ones((height, width, 3), dtype=np.uint8) * [50, 0, 0]
red2_image = np.ones((height, width, 3), dtype=np.uint8) * [100, 0, 0]
red3_image = np.ones((height, width, 3), dtype=np.uint8) * [180, 0, 0]
red4_image = np.ones((height, width, 3), dtype=np.uint8) * [240, 0, 0]
red5_image = np.ones((height, width, 3), dtype=np.uint8) * [255, 0, 0]
# Create green images with varying intensities
green_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 50, 0]
green1_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 100, 0]
green2_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 180, 0]
green3_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 340, 0]
green4_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 255, 0]
# Create blue images with varying intensities
blue1_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 0, 50]
blue2_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 0, 100]
blue3_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 0, 180]
blue4_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 0, 240]
blue5_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 0, 255]
# Create yellow images with varying intensities
rg1_image = np.ones((height, width, 3), dtype=np.uint8) * [50, 50, 0]
rg2_image = np.ones((height, width, 3), dtype=np.uint8) * [100, 100, 0]
rg3_image = np.ones((height, width, 3), dtype=np.uint8) * [180, 180, 0]
rg4_image = np.ones((height, width, 3), dtype=np.uint8) * [240, 240, 0]
rg5_image = np.ones((height, width, 3), dtype=np.uint8) * [255, 255, 0]
# Create cyan images with varying intensities
gb1_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 50, 50]
gb2_image = np.ones((height, width, 3), dtype=np.uint8) * [0,  100, 100]
gb3_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 180, 180]
gb4_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 240, 240]
gb5_image = np.ones((height, width, 3), dtype=np.uint8) * [0, 255, 255]
#   Create magenta images with varying intensities
rb1_image = np.ones((height, width, 3), dtype=np.uint8) * [50, 0, 50]
rb2_image = np.ones((height, width, 3), dtype=np.uint8) * [100, 0, 100]
rb3_image = np.ones((height, width, 3), dtype=np.uint8) * [180, 0, 180]
rb4_image = np.ones((height, width, 3), dtype=np.uint8) * [240, 0, 240]
rb5_image = np.ones((height, width, 3), dtype=np.uint8) * [255, 0, 255]  


# All images in a list
images = [black_image, gray1_image, gray2_image, gray3_image,
          white_image, red1_image,red2_image,red3_image,red4_image,red5_image,
          green_image,green1_image,green2_image,green3_image,green4_image,                   
          blue1_image,blue2_image,blue3_image,blue4_image,blue5_image,rg1_image,
          rg2_image,rg3_image,rg4_image,rg5_image,gb1_image,gb2_image,
          gb3_image,gb4_image,gb5_image,rb1_image,rb2_image,rb3_image,
          rb4_image,rb5_image]

# Plot with 5 columns (7 rows)
plt.figure(figsize=(15, 8))

for i in range(35):
    plt.subplot(7, 5, i + 1)  # 7 rows, 5 columns
    plt.imshow(images[i])
    #plt.axis('off')

plt.tight_layout()
plt.show()
