import os
from tqdm import tqdm
from utils_metrics import compute_mIoU, show_results
import glob
import numpy as np
import torch
import cv2
from unetpp_model2 import Unetplusplus
from torchvision.transforms.functional import resize
from PIL import Image


def cal_miou(test_dir="E:/finalproject/ACDC_dataset/ACDC_MYO/ACDC_1/Testing_Images",
             pred_dir="E:/finalproject/ACDC_dataset/ACDC_MYO/ACDC_1/Results",
             gt_dir="E:/finalproject/ACDC_dataset/ACDC_MYO/ACDC_1/Testing_Labels"):
    # ---------------------------------------------------------------------------#
    #   miou_mode as a flag
    #   miou_mode = 0 : both the predicted images and miou
    #   miou_mode = 1 : only get the predicted image
    #   miou_mode = 2 : calculate the miou
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    num_classes = 2
    name_classes = ["background", "LV"]

    #load the model
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = Unetplusplus(num_classes=1, input_channels=1)
        net.to(device=device)

        # load
        net.load_state_dict(torch.load('unetplusplus_model_MYO_1_2.pth', map_location=device))

        # test
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".png")
            img = cv2.imread(image_path)
            origin_shape = img.shape
            # print(origin_shape)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (512, 512))
            img = img.reshape(1, 1, img.shape[0], img.shape[1])

            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)

            # predict
            pred = net(img_tensor)
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou and recall.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,name_classes)
        print("IoUs:",IoUs)
        print("recall:",PA_Recall)

        # save the evaluation results
        miou_out_path = "20240313_MYO1_2_test_results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    cal_miou()
