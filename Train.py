import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
import numpy as np
from dataset import Mydataset
import torch


def train(images_trian, target_train, images_test, target_test):
    train_dataset = Mydataset(images_trian, target_train)
    test_dataset = Mydataset(images_test, target_test)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=4)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=4)
    # get the num of the different classes
    num = len(np.unique(target_train))

    # prehandle with data , to avoid error in predict ,make all of tag in common
    if min(target_train) == 0:
        pass
    else:
        target_train -= 1
        target_test -= 1

    class_names = 2
    print('class_names:{}'.format(num))
    print("shape of trainset and testset{}_{}".format(images_trian.shape, images_test.shape))
    # 'cuda:0' if torch.cuda.is_available() else
    # 训练设备 CPU/GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("train_device:{}".format(device.type))

    # 随机显示一个batch
    # plt.figure()
    # utils.imshow(next(iter(train_dataloader)))
    # plt.show()

    # -------------------------模型选择，优化方法， 学习率策略----------------------
    model = models.resnet18(pretrained=True)

    # 全连接层的输入通道in_channels个数
    num_fc_in = model.fc.in_features

    # 改变全连接层，2分类问题，out_features=2

    model.fc = nn.Linear(num_fc_in, num)

    # 模型迁移到CPU/GPU
    model = model.to(device)

    # 定义损失函数
    loss_fc = nn.CrossEntropyLoss()

    # 选择性优化方法
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # 学习率调整策略
    # 每10个step调整一次
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)  # step_size

    # ----------------训练过程-----------------
    num_epochs = 200

    accuracy_list = []
    for epoch in range(num_epochs):

        running_loss = 0.0
        exp_lr_scheduler.step()

        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0]
            labels = sample_batch[1]
            inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1).to(torch.float32)
            lables = labels
            model.train()

            # GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # foward
            outputs = model(inputs)

            # loss
            loss = loss_fc(outputs, labels)
            # loss求导，反向
            loss.backward()

            # 优化
            optimizer.step()

            #
            running_loss += loss.item()

            # 測試

        if epoch % 10 == 9:
            correct = 0
            total = 0
            model.eval()
            for images_test, labels_test in val_dataloader:
                images_test = images_test.to(device).unsqueeze(1).repeat(1, 3, 1, 1).to(torch.float32)
                labels_test = labels_test.to(device)
                outputs_test = model(images_test)
                _, prediction = torch.max(outputs_test, 1)
                correct += (torch.sum((prediction == labels_test))).item()
                # print(prediction, labels_test, correct)
                total += labels_test.size(0)
            print('[{}, {}] running_loss = {:.5f} accurcay = {:.5f}'.format(epoch + 1, i + 1, running_loss / 20,
                                                                            correct / total))
            accuracy_list.append(correct / total)


    print(f'best accuracy{max(accuracy_list)}')

    print('training finish !')
    return max(accuracy_list)
