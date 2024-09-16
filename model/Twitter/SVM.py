import sys, os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import sys, os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Process.rand5fold import *
from tools.evaluate import *
from tqdm import tqdm

# 列表存储每个批次的损失和准确率
batch_train_accs = []
batch_val_accs = []

# 定义简单的SVM模型
class SimpleSVM:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

def process_data_for_svm(data_list):
    """
    将图数据转换为适合SVM的数据格式：将每个节点的特征展平为二维矩阵。
    假设 `data_list` 是图数据列表，其中每个元素 `data` 都有 `x` 和 `y` 属性，
    分别表示节点的特征和标签。
    """
    X = []
    y = []
    for data in data_list:
        X.append(data.x.cpu().numpy().flatten())  # 将节点特征展平
        y.append(data.y.cpu().numpy())  # 标签
    X = np.array(X)
    y = np.array(y).flatten()  # 将标签展平为一维
    return X, y

def train_svm(treeDic, x_test, x_train, kernel, C, batchsize, dataname, iter):
    # 处理训练集和测试集数据为适合SVM的格式
    traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, 0, 0)
    X_train, y_train = process_data_for_svm(traindata_list)
    X_test, y_test = process_data_for_svm(testdata_list)

    # 初始化SVM模型
    model = SimpleSVM(kernel=kernel, C=C)

    # 训练SVM
    model.train(X_train, y_train)

    # 预测训练集和测试集
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # 计算训练集和测试集的准确率
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_test, test_preds)

    # 记录训练和验证的准确率
    batch_train_accs.append(train_acc)
    batch_val_accs.append(val_acc)

    print(f"Iter {iter:03d} | Train_Accuracy: {train_acc:.4f} | Val_Accuracy: {val_acc:.4f}")
    
    return train_acc, val_acc

# 超参数和数据集设置
kernel = 'linear'
C = 1.0
batchsize = 16
datasetname = "Weibo"
iterations = int(sys.argv[1])  # 从命令行获取迭代次数
test_accs = []

# 训练多个迭代
for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train,  \
    fold3_x_test, fold3_x_train,  \
    fold4_x_test, fold4_x_train = load5foldData(datasetname)
    
    treeDic = loadTree(datasetname)
    
    train_acc, val_acc = train_svm(treeDic, fold0_x_test, fold0_x_train, kernel, C, batchsize, datasetname, iter)
    test_accs.append(val_acc)

# 打印最终结果
print("weibo: | Total_Test_Accuracy: {:.4f}".format(np.mean(test_accs)))
