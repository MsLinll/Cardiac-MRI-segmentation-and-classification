The project implemented the semi-supervised image segmentation based on the U-Net++ network model, and extracted the feature of the segmented image and predicted the heart disease. Finally, the classification accuracy of 91.67% was obtained

Readacdc.py is used to read the initial cardiac MRI slice.
dataset.py is applied for input and read the images to the training and testing network.
enhancement.py is to do the image enhancment task by rotation and cutting.

There are U-Net++ network model codes for segmentation, including training, testing, prediction, and evaluation.

extract_features.py is the code for extract the cardiac features from the segmented images.

Two classifiers,SVM and random forest, are introduced.  -----------------svm3.py     random_forest.py
The trained model of random forest is in the random_forest.pth file.