# 取一张图片进行识别
import torch
from train import *
import glob
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torchvision
from skimage import io,transform
import matplotlib.pyplot as plt

if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.load_state_dict(torch.load('./MNIST_SD.pth'))  # 加载模型
    model = torch.load('./MNIST.pth') #加载模型
    model = model.to(device)
    model.eval()    #把模型转为test模式

    img = cv2.imread('./img/33.jpg', 0)  #以灰度图的方式读取要预测的图片
    img = cv2.resize(img, (28, 28))

    # cv2.imshow("frame",img)

    height,width=img.shape
    dst=np.zeros((height,width),np.uint8)
    for i in range(height):
        for j in range(width):
            dst[i,j]=255-img[i,j]

    img = dst
    # cv2.imshow("frame", img)

    img=np.array(img).astype(np.float32)
    img=np.expand_dims(img,0)
    img=np.expand_dims(img,0)#扩展后，为[1，1，28，28]
    img=torch.from_numpy(img)
    img = img.to(device)
    output=model(Variable(img))
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(prob)  #prob是10个分类的概率
    pred = np.argmax(prob) #选出概率最大的一个
    print(pred.item())


