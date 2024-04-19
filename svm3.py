from sklearn import svm
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

# 当使用numpy中的loadtxt函数导入该数据集时，假设数据类型dtype为浮点型，但是很明显数据集的第五列的数据类型是字符串并不是浮点型。
# 因此需要额外做一个工作，即通过loadtxt()函数中的converters参数将第五列通过转换函数映射成浮点类型的数据。
# 首先，我们要写出一个转换函数：
# 定义一个函数，将不同类别标签与数字相对应
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def heart_type(s):
    class_label = {b'0':0, b'1':1}
    return class_label[s]

# （1）使用numpy中的loadtxt读入数据文件
filepath = 'features_enhancement2.txt'  # 数据文件路径
data = np.loadtxt(filepath, dtype = float, delimiter=',', converters={0: heart_type})
# 以上4个参数中分别表示：
# filepath ：文件路径。eg：C:/Dataset/iris.txt。
# dtype=float ：数据类型。eg：float、str等。
# delimiter=',' ：数据以什么分割符号分割。eg：‘，’。
# converters={4:iris_type} ：对某一列数据（第四列）进行某种类型的转换，将数据列与转换函数进行映射的字典。eg：{1:fun}，含义是将第2列对应转换函数进行转换。converters={4: iris_type}中“4”指的是第5列。


# （2）将原始数据集划分成训练集和测试集
y, X = np.split(data, (1,),axis=1)
# np.split 按照列（axis=1）进行分割，从第四列开始往后的作为y 数据，之前的作为X 数据。函数 split(数据，分割位置，轴=1（水平分割） or 0（垂直分割）)。
x = X[:, 0:4]  # 在 X中取前两列作为特征（为了后期的可视化画图更加直观，故只取前两列特征值向量进行训练）
#print("x特征：",x)
#print("y标签：",y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.2)
print(len(x_train), len(y_train))
print(x_train.shape, y_train.shape)

# 用train_test_split将数据随机分为训练集和测试集，测试集占总数据的30%（test_size=0.3),random_state是随机数种子
# 参数解释：
# x：train_data：所要划分的样本特征集。
# y：train_target：所要划分的样本结果。
# test_size：样本占比，如果是整数的话就是样本的数量。
# random_state：是随机数的种子。
# （随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
# 随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。）


# （3）搭建模型，训练SVM分类器
# classifier=svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=0.1)
# kernel='linear'时，为线性核函数，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
classifier = svm.SVC(kernel='rbf', gamma=0.005, decision_function_shape='ovo', C=0.8)
# kernel='rbf'（default）时，为高斯核函数，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# decision_function_shape='ovo'时，为one v one分类问题，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
# decision_function_shape='ovr'时，为one v rest分类问题，即一个类别与其他类别进行划分。
# 开始训练
classifier.fit(x_train, y_train.ravel())

# 调用ravel()函数将矩阵转变成一维数组
# （ravel()函数与flatten()的区别）
# 两者所要实现的功能是一致的（将多维数组降为一维），
# 两者的区别在于返回拷贝（copy）还是返回视图（view），
# numpy.flatten() 返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
# 而numpy.ravel()返回的是视图（view），会影响（reflects）原始矩阵。

def show_accuracy(y_hat, y_train, str):
    pass

# （4）计算svm分类器的准确率
print("SVM-输出训练集的准确率为：", classifier.score(x_train, y_train))
y_hat = classifier.predict(x_train)
show_accuracy(y_hat, y_train, '训练集')
print("SVM-输出测试集的准确率为：", classifier.score(x_test, y_test))
y_hat = classifier.predict(x_test)
show_accuracy(y_hat, y_test, '测试集')

torch.save(classifier,"svm1.pth")
print(classifier)

prediction = classifier.predict(x_test)
confusion_matrix = confusion_matrix(y_test,prediction)
print(confusion_matrix)

matrix = ConfusionMatrixDisplay.from_predictions(y_test, prediction)
matrix.plot(cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.show()





