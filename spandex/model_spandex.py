import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import pandas as pd
import argparse
import os

#获取命令行传参
parser = argparse.ArgumentParser(description='test module of argparse')
parser.add_argument(
    '-p', '--path', type=str,
)
args = parser.parse_args()
path = args.path

#获取显卡
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载模型
model_save_path = 'checkpoint.pth.tar'
checkpoint = torch.load(model_save_path)
model_conv = checkpoint['model']
model_conv.eval()

#设置图片预处理
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

#打开图片并做处理
image_PIL = Image.open(path)
image_tensor = test_transforms(image_PIL)
image_tensor.unsqueeze_(0)
image_tensor = image_tensor.to(DEVICE)

#传入模型预测
out = model_conv(image_tensor)
pred = out.max(1)[1]
res = pred.item()
cls_map = {0:'0/0',1:'1/1',2:'0/1'}
print(cls_map[res])


