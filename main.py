import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
import numpy as np
from dataset import  Mydataset
from pyts import datasets
from T2I import generate_transform
from Train import  train

def main(dataset_name):
    dataset_name = "Coffee"
    # 获取四种变换形式
    gasf,gadf,RcP,Mark = generate_transform()
    #获取需要训练二点数据集
    (data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset(dataset_name,return_X_y=True)
    #def train needs data_train,data_test,target_train,target_test
    data_train = gasf(data_train)
    target_train = gasf(target_train)

    best_result = train(data_train, data_test, target_train, target_test)
    print(f"{dataset_name} {best_result}")

if __name__ =="__main__":
    if __name__ == '__main__':
        main("Coffee")