import torch
import torch.nn.functional as Fun
import pandas as pd

data = pd.read_csv(r".\dataset/labeled_shffle.csv", header=None)

Fearture = data.drop([4], axis=1).values
label = data[4].values

input = torch.FloatTensor(Fearture)
label = torch.LongTensor(label)


# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x


# 目前最高： hidden = 200 ， lr=0.0001
net = Net(n_feature=4, n_hidden=180, n_output=2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
# SGD:随机梯度下降法
loss_func = torch.nn.CrossEntropyLoss()
# 设定损失函数

# 开始训练
for epoch in range(100000):
    out = net(input)
    loss = loss_func(out, label)
    # 输出与label对比
    optimizer.zero_grad()
    # 初始化
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 999:
        y_pred_label = torch.where(out >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        acc = torch.eq(y_pred_label, label).sum().item() / label.size(0)
        print("loss = ", loss.item(), "acc = ", acc)

out = net(input)
# out是一个计算矩阵
prediction = torch.max(out, 1)[1]
pred_y = prediction.numpy()
print('预测结果：', pred_y)
# 预测y输出数列
target_y = label.data.numpy()
# 实际y输出数据
print('标签：', target_y)

# 计算准确率——自己写的
L = pred_y.shape[0]
flag = 0
for i in range(L):
    if pred_y[i] == target_y[i]:
        flag += 1
print("bp准确率为：", flag / L)
