from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def heart_type(s):
    class_label = {b'0': 0, b'1': 1}
    return class_label[s]

filepath = 'features_enhancement2.txt'
data = np.loadtxt(filepath, dtype=float, delimiter=',', converters={0: heart_type})

# Divide the training set and test set
y, X = np.split(data, (1,), axis=1)
x = X[:, 0:6]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.2)

# set the random forest classifier parameters
classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=1 ,random_state=42)
classifier.fit(x_train, y_train.ravel())

def show_accuracy(y_hat, y_train, str):
    pass


print("accuracy on the training set：", classifier.score(x_train, y_train))
y_hat = classifier.predict(x_train)
show_accuracy(y_hat, y_train, 'training set')
print("accuracy on the testing set：", classifier.score(x_test, y_test))
y_hat = classifier.predict(x_test)
show_accuracy(y_hat, y_test, 'testing set')

# save the classifier model
torch.save(classifier, "random_forest.pth")
print(classifier)

# Draw the confusion matrix
prediction = classifier.predict(x_test)
conf_matrix = confusion_matrix(y_test, prediction)
print(conf_matrix)

matrix = ConfusionMatrixDisplay(conf_matrix, display_labels=['Class 0', 'Class 1'])
matrix.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
