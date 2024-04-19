import numpy as np
import torch
import os
import cv2
import glob
from unetpp_model2 import Unetplusplus

def monte_carlo_predict(model,img_tensor,num_iteration=100):
    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_iteration):
            output = model(img_tensor)
            predictions.append(output.cpu().numpy())

    return np.array(predictions)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:',device)

    net = Unetplusplus(num_classes=1, input_channels=1,supervised=False)
    net.to(device=device)
    net.load_state_dict(torch.load('unetplusplus_model_1_8.pth', map_location=device))

    # test
    net.eval()
    tests_path = glob.glob('F:/finalproject/ACDC_dataset/ACDC_LV/enhancement/LV_1/0_enhancement/*.png')
    uncertainties = []

    for test_path in tests_path:

        save_res_path = test_path.split('.')[0] + '_predict.png'
        img = cv2.imread(test_path)
        origin_shape = img.shape
        #print(origin_shape)

        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_g = cv2.resize(img_g, (512, 512))
        img_g = img_g.reshape(1, 1, img_g.shape[0], img_g.shape[1])

        img_tensor = torch.from_numpy(img_g)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # predict
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # save the results
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_res_path, pred)

        # uncertainty
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        uncertainty_predictions = monte_carlo_predict(net, img_tensor)
        uncertainty = np.mean(np.var(uncertainty_predictions,axis=0))
        uncertainties.append(uncertainty)
        file_name = os.path.basename(test_path)
        print(f"{file_name} -----  uncertainty:",uncertainty)

    uncertainties = np.array(uncertainties)
    uncertainty_mean = np.mean(uncertainties)
    uncertainty_std = np.std(uncertainties)
    print("mean uncertainty:",uncertainty_mean)
    print("uncertainty standard deviation:",uncertainty_std)

    threshold_1 = uncertainty_mean - 0.3*uncertainty_std
    threshold_2 = uncertainty_mean - 1.2*uncertainty_std
    print("Threshold of uncertainty:",threshold_1)

# expand the training datasets
    training_image_path = 'E:/finalproject/20240228ACDC_2_5/Training_Images'
    training_label_path = 'E:/finalproject/20240228ACDC_2_5/Training_Labels'
    for i, test_path in enumerate(tests_path):
        if uncertainties[i] > threshold_2 and uncertainties[i] < threshold_1:
            print(test_path,'allow to join the training dataset')
            file_name = os.path.basename(test_path)
            image_save_path = os.path.join(training_image_path,file_name)
            img = cv2.imread(test_path)
            cv2.imwrite(image_save_path,img)
            # 保存标签
            pred_path = test_path.split('.')[0] + '_predict.png'
            label_save_path = os.path.join(training_label_path, file_name)
            pred_img = cv2.imread(pred_path)
            cv2.imwrite(label_save_path,pred_img)
        else:
            print(test_path,'not allowed to join the training dataset')

    print("The predicted images have been saved successfully!")