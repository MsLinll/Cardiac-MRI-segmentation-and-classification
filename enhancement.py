import os
import cv2
import numpy as np
import random

def augment_image(image):
    augmented_images = []
    for _ in range(27):
        # Rotation of image
        angle = random.randint(-30, 30)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))

        # image cutting
        start_x = random.randint(0, cols // 4)  # Randomly select the coordinates of the clipping start point
        start_y = random.randint(0, rows // 4)
        end_x = random.randint(cols * 3 // 4, cols)  # Randomly select the coordinates of the clipping end point
        end_y = random.randint(rows * 3 // 4, rows)
        cropped_image = rotated_image[start_y:end_y, start_x:end_x]

        augmented_images.append(cropped_image)
    return augmented_images


folder_path = 'F:/finalproject/ACDC_dataset/abnormal_enhancement/2'

for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        image = cv2.imread(os.path.join(folder_path, filename))

        augmented_images = augment_image(image)

        # save the augmented images
        for i, augmented_image in enumerate(augmented_images):
            save_path = 'F:/finalproject/ACDC_dataset/abnormal_enhancement/2_enhanced'
            cv2.imwrite(os.path.join(save_path, f'{filename[:-4]}_{i}.png'), augmented_image)
