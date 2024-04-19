import torch
import monai
import numpy as np
import matplotlib.pyplot as plt
import glob

datapath = r'F:/finalproject/unetpp_load/database/training'
print("successfully",datapath)
transform = monai.transforms.Compose([monai.transforms.LoadImageD(("image", "label")),
                                      monai.transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=5, upper=95, b_min=0, b_max=1, clip=True),
                                      monai.transforms.RandGaussianNoised(("image"), prob=1, std=.3) ])
#transform = monai.transforms.LoadImageD(("image", "label"))
file_dict = {"image": "{}/patient001/patient001_frame01.nii.gz".format(datapath),
             "label": "{}/patient001/patient001_frame01_gt.nii.gz".format(datapath)}
data_dict = transform(file_dict)


def visualize_data(pt_dict, batch=False):
    image = pt_dict["image"].squeeze()
    label = pt_dict["label"].squeeze()
    if batch:
        image = image.permute((1, 2, 0))
        label = label.permute((1, 2, 0))
    plt.figure(figsize=(20, 20))
    num_slices = image.shape[2]
    num_rows_cols = int(np.ceil(np.sqrt(num_slices)))
    for z in range(num_slices):
        plt.subplot(num_rows_cols, num_rows_cols, 1 + z)
        plt.imshow(image[:, :, z], cmap='gray')
        plt.axis('off')
        plt.imshow(np.ma.masked_where(label[:, :, z] != 2, label[:, :, z] == 2), alpha=0.6, cmap='Blues', clim=(0, 1))
        plt.imshow(np.ma.masked_where(label[:, :, z] != 3, label[:, :, z] == 3), alpha=0.6, cmap='Greens', clim=(0, 1))
        plt.imshow(np.ma.masked_where(label[:, :, z] != 1, label[:, :, z] == 1), alpha=0.6, cmap='Reds', clim=(0, 1))
        plt.title('Slice {}'.format(z + 1))
    plt.show()

visualize_data(data_dict)

data_dict_transformed = transform(file_dict)
visualize_data(data_dict_transformed)