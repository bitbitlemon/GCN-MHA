# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

cwd = os.getcwd()

# 检查并创建所需的目录
data_dir = os.path.join(cwd, 'data/Weibo')
graph_dir = os.path.join(cwd, 'data/Weibograph')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"创建目录：{data_dir}")

if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    print(f"创建目录：{graph_dir}")

# 检查并创建所需的文件
treePath = os.path.join(data_dir, 'weibotree.txt')
labelPath = os.path.join(data_dir, 'weibo_id_label.txt')

if not os.path.exists(treePath):
    with open(treePath, 'w', encoding='utf-8') as f:
        f.write('')  # 可以在这里写入默认内容
    print(f"创建文件：{treePath}")

if not os.path.exists(labelPath):
    with open(labelPath, 'w', encoding='utf-8') as f:
        f.write('')  # 可以在这里写入默认内容
    print(f"创建文件：{labelPath}")

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq = float(pair.split(':')[1])
        index = int(pair.split(':')[0])
        if index <= 5000:
            wordFreq.append(freq)
            wordIndex.append(index - 1)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
            rootindex = indexC - 1
            root_index = nodeC.index
            root_word = nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index) > 0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    ## 3. convert tree to matrix and edgematrix
    matrix = np.zeros([len(index2node), len(index2node)])
    raw = []
    col = []
    x_word = []
    x_index = []
    edgematrix = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i + 1].children != None and index2node[index_j + 1] in index2node[index_i + 1].children:
                matrix[index_i][index_j] = 1
                raw.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i + 1].word)
        x_index.append(index2node[index_i + 1].index)
    edgematrix.append(raw)
    edgematrix.append(col)
    return x_word, x_index, edgematrix, rootfeat, rootindex

def getfeature(x_word, x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i]) > 0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def main():
    treePath = os.path.join(cwd, 'data/Weibo/weibotree.txt')
    print("reading Weibo tree")
    treeDic = {}
    for line in open(treePath, encoding='utf-8'):
        line = line.rstrip()
        parts = line.split('\t')
        if len(parts) < 4:
            continue  # 跳过不完整的行
        eid, indexP, indexC_str, Vec = parts[0], parts[1], parts[2], parts[3]
        try:
            indexC = int(indexC_str)
        except ValueError:
            continue  # 跳过无法转换为整数的索引
        if eid not in treeDic:
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    print('tree no:', len(treeDic))

    labelPath = os.path.join(cwd, "data/Weibo/weibo_id_label.txt")
    print("loading weibo label:")
    event, y = [], []
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath, encoding='utf-8'):
        line = line.rstrip()
        parts = line.split(' ')
        if len(parts) < 2:
            continue  # 跳过不完整的行
        eid, label_str = parts[0], parts[1]
        try:
            label = int(label_str)
        except ValueError:
            continue  # 跳过无法转换为整数的标签
        labelDic[eid] = label
        y.append(labelDic[eid])
        event.append(eid)
        if labelDic[eid] == 0:
            l1 += 1
        if labelDic[eid] == 1:
            l2 += 1

    print(len(labelDic), len(event), len(y))
    print(l1, l2)

    def loadEid(event, id, y):
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event) > 1:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
            x_x = getfeature(x_word, x_index)
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
            np.savez(os.path.join(graph_dir, id + '.npz'), x=x_x, root=rootfeat, edgeindex=tree, rootindex=rootindex,
                     y=y)
            return None
        x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
        x_x = getfeature(x_word, x_index)
        return rootfeat, tree, x_x, [rootindex]

    print("loading dataset")
    results = Parallel(n_jobs=30, backend='threading')(
        delayed(loadEid)(treeDic[eid] if eid in treeDic else None, eid, labelDic[eid]) for eid in tqdm(event))
    return

if __name__ == '__main__':
    main()
