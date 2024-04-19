import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        #Read all images under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Training_Images/*.png'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        image_path = self.imgs_path[index]

        # Generate label_path from image_path
        label_path = image_path.replace('Training_Images', 'Training_Labels')
        label_path = label_path.replace('.png', '.png')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        if label.max() > 1:
            label = label / 255

        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
       # print('successfuly read the images and labels!')
       # print('image size:',image.shape)    #(1,512,512)
       # print('label size:',label.shape)    #(1,512,512)

        return image, label

    def __len__(self):

        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("E:/finalproject/ACDC_1/")
    print("The number of dataset：", len(isbi_dataset))   #training datasets：600 images
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,batch_size=1, shuffle=True)

    for image, label in train_loader:
        print(image.shape)
        print(label.shape)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(image.squeeze(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Label')
        plt.imshow(label.squeeze(), cmap='gray')
        plt.show()
