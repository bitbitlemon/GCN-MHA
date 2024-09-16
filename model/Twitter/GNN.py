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
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv  # 引入其他GNN层
import copy

# 定义TD方向的GNN
class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, gnn_type='GCN'):
        super(TDrumorGCN, self).__init__()
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(in_feats, hid_feats)
            self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        elif gnn_type == 'GraphSAGE':
            self.conv1 = GraphSAGE(in_feats, hid_feats)
            self.conv2 = GraphSAGE(hid_feats + in_feats, out_feats)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(in_feats, hid_feats, heads=8, concat=False)
            self.conv2 = GATConv(hid_feats + in_feats, out_feats, heads=8, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

# 定义BU方向的GNN
class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, gnn_type='GCN'):
        super(BUrumorGCN, self).__init__()
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(in_feats, hid_feats)
            self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        elif gnn_type == 'GraphSAGE':
            self.conv1 = GraphSAGE(in_feats, hid_feats)
            self.conv2 = GraphSAGE(hid_feats + in_feats, out_feats)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(in_feats, hid_feats, heads=8, concat=False)
            self.conv2 = GATConv(hid_feats + in_feats, out_feats, heads=8, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

# 整合TD和BU的网络
class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, gnn_type='GCN'):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, gnn_type)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, gnn_type)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 4)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# 列表存储每个批次的损失和准确率
batch_train_losses = []
batch_train_accs = []
batch_val_losses = []
batch_val_accs = []

def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, dataname, iter, gnn_type='GCN'):
    model = Net(5000, 64, 64, gnn_type).to(device)
    BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    optimizer = th.optim.Adam([
        {'params': base_params},
        {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr / 5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr / 5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)

        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels = model(Batch_data)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)

            # 记录每个批次的训练损失和准确率
            batch_train_losses.append(loss.item())
            batch_train_accs.append(train_acc)

            print("Iter {:03d} | Epoch {:05d} | Batch {:02d} | Train_Loss {:.4f} | Train_Accuracy {:.4f}".format(
                iter, epoch, batch_idx, loss.item(), train_acc))
            batch_idx += 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_F1, temp_val_F2, temp_val_F3, temp_val_F4 = [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)

        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            temp_val_accs.append(val_acc)

            # 假设 evaluation4class 函数计算并返回这些指标
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_F1.append(F1)
            temp_val_F2.append(F2)
            temp_val_F3.append(F3)
            temp_val_F4.append(F4)

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))

        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses), np.mean(temp_val_accs)))

        # 调用 early_stopping，提供所有必需的参数
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses, train_accs, val_accs

# 设置训练参数
lr = 0.0005
weight_decay = 1e-4
patience = 10
n_epochs = 50
batchsize = 128
TDdroprate = 0.2
BUdroprate = 0.2
datasetname = sys.argv[1]  # "Twitter15" or "Twitter16"
iterations = int(sys.argv[2])
gnn_type = sys.argv[3]  # "GCN", "GraphSAGE" or "GAT"
model = "GCN"
device = th.device('cpu')
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []

for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test, fold4_x_train = load5foldData(datasetname)
    
    treeDic = loadTree(datasetname)
    
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(
        treeDic, fold0_x_test, fold0_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter, gnn_type)
    
    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(
        treeDic, fold1_x_test, fold1_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter, gnn_type)
    
    train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(
        treeDic, fold2_x_test, fold2_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter, gnn_type)
    
    train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(
        treeDic, fold3_x_test, fold3_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter, gnn_type)
    
    train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(
        treeDic, fold4_x_test, fold4_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter, gnn_type)
    
    test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
    NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)

print("Total_Test_Accuracy: {:.4f} | NR F1: {:.4f} | FR F1: {:.4f} | TR F1: {:.4f} | UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
