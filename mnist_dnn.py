# 基于MNIST数据集实现简单深度神经网络
# 由1个输入层、1个全连接层结构的隐含层和1个输出层构建
# 《Pytorch机器学习从入门到实战》 Page：89


# 【1】配置库和配置参数
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import time

print('GPU：'+ format(torch.cuda.is_available()))

# hyper parameters 配置参数
torch.manual_seed(1)  # 设置随机数种子
input_size = 784      # 即28*28
hidden_size = 500
num_classes = 100
num_epochs = 5        # 训练次数
batch_size = 100      # 批处理大小
learning_rate = 0.001  # 学习率

# 【2】加载MNIST
# MNIST dataset下载训练集MNIST手写数字训练集
train_dataset = dsets.MNIST(root='./data',    # 数据保持的位置
                            train = True,      # 训练集
                            transform = transforms.ToTensor(), # 一个取值范围是[0,255]的PIL.Image
                                                               # 转化为取值范围是[0,1.0]的torch.FloadTensor
                            download = True    # 下载数据
                            )
test_dataset = dsets.MNIST(root='./data',
                           train = False,      # 测试集
                           transform = transforms.ToTensor()
                           )

# 【3】数据的批处理
# Data Loader （Input Pipeline）
# 数据的批处理，尺寸大小为batch.size, 在训练集中，shuffle必须设置为True，表示次序是随机的
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False)

# 【4】创建DNN模型
# Neural Network Model（1 hidden layer） 定义神经网络模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):   # 在构造函数_init__中实现层的参数定义
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 分别是输入/输出的二维张量的大小；后者也代表全连接层神经元个数
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # self.relu1 = nn.ReLU()
        # self.fc3 = nn.Linear(num_classes,10)

    def forward(self, x):               # 在前向传播forward函数里面实现前向运算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        # out = self.relu(out)
        # out = self.fc3(out)
        return out

# 打印模型，呈现网络结构
net = Net(input_size, hidden_size, num_classes)
print(net)

# 【5】 训练流程，将images、labels都用Variable包起来放入模型中计算输出，最后计算Loss和正确率
# Loss and Optimizer 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

start_time = time.time()
# Train the Model 开始训练
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):          # 批处理
        # convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # Forward and Backward and Optmize
        optimizer.zero_grad()             # zero the gradient buffer 梯度清零，以免影响其他batch
        outputs = net.forward(images)             # 前向传播
        loss = criterion(outputs, labels)    # loss
        loss.backward()                   # 前向传播，计算梯度
        optimizer.step()                  # 梯度更新

        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%
                  (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item())
                  )

end_time = time.time()
print('训练用时：{:.5f}s'.format(end_time - start_time))
# torch.save(net.state_dict(), "dnn_mnist_parameter.pth")  # 保存模型参数

# 【6】 在测试集测试识别率
# Test the Model 测试集上验证模型
correct = 0
total = 0
for images,labels in test_loader:               # test set 批处理
    images = Variable(images.view(-1, 28*28))
    outputs = net.forward(images)
    _, predicted = torch.max(outputs.data, 1)     # 预测结果
    total += labels.size(0)                       # 正确结果
    correct += (predicted == labels).sum().item()       # 正确结果总数
print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct/total))


