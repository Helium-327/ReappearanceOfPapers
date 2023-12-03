import json
import os
import sys

import torch
import torchvision
import torch.nn as nn
from model import LeNet, AlexNet, vgg, GoogleNet
import torch.optim as optim
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""超参数"""
batch_size = 128
epochs = 10
num_classes = 5
init_weights = True
# TODO: 更换模型时，需修改模型名称和训练权重的保存地址
model_name = GoogleNet
# save_path = f'./models/VGGNet.pth'
save_path = f'../models/{model_name.__name__}.pth'
num_workers = 0


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    """数据预处理"""
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪 224 * 224
                                     transforms.RandomHorizontalFlip(),  # 随机翻转
                                     transforms.ToTensor(),  # 转换成Tensor
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 标准化
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # """准备数据（方式一:下载）"""
    # # 内置下载
    # ## 训练集
    # train_dataset = torchvision.datasets.CIFAR10(root='./data',
    #                                          train=True,
    #                                          download=True,
    #                                          transform=data_transform)
    # ## 验证集
    # val_dataset = torchvision.datasets.CIFAR10(root='./data',
    #                                        train=False,
    #                                        download=False,
    #                                        transform=data_transform)
    # ## 定义类型元组
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    """准备数据（方式2：本地读取）"""
    data_root = os.path.abspath(os.path.join(os.getcwd()))  # os.getcwd()是Python中的一个函数，用于获取当前工作目录
    image_path = os.path.join(data_root, "../data", "flower_data")
    assert os.path.exists(image_path), f"{image_path} path does not exist."  # 异常提示
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform['val'])

    flower_list = train_dataset.class_to_idx
    # print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 把类别字典写入json文件
    json_str = json.dumps(cla_dict, indent=4)  # json 文件的操作
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    print(f'Using {num_workers} dataloader workers every process')

    """加载数据"""
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             num_workers=num_workers)
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(f'Using {train_num} images for training, {val_num} images for validation.')

    # # ================================================================================
    # # 测试观察
    # test_data_iter = iter(val_loader)
    # test_image, test_label = next(test_data_iter)
    #
    # def imshow(img):
    #     """Tensor To IMG"""
    #     img = img / 2 + 0.5
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # (C * H * W) ----> (H * W * C)
    #     plt.show()
    #
    # print(''.join(f'{cla_dict[test_label[j].item()]:5s}' for j in range(4)))
    # imshow(utils.make_grid(test_image))
    # # ==================================================================================

    """定义网络模型"""
    # TODO:不同的模型，定义方式会有差异，记得更改
    model = model_name(num_classes=num_classes, init_weights=init_weights)
    # model = vgg(model_number=model_name, num_classes=num_classes, init_weights=init_weights)
    model.to(device)

    """定义损失函数"""
    loss_function = nn.CrossEntropyLoss()

    """定义优化器"""
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    train_steps = len(train_loader)
    """模型开始训练"""

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)  # 定义训练进度条
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # TODO:不同模型的输出形式也不同
            # outputs = model(images.to(device))
            logits, aux_logits2, aux_logits1 = model(images.to(device))
            loss0 = loss_function(logits, labels.to(device))  # 计算损失
            loss1 = loss_function(aux_logits1, labels.to(device))  # 计算损失
            loss2 = loss_function(aux_logits2, labels.to(device))  # 计算损失
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            running_loss += loss.item()  # 记录损失值

            train_bar.desc = f"Train epoch[{epoch + 1} / {epochs}], loss:{loss:.3f}"

        # val
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_dataset in val_bar:
                val_images, val_labels = val_dataset
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print(f'Val   epoch[{epoch + 1} / {epochs}] \t train_loss: {running_loss / train_steps:.3f} val_accuracy: {val_accurate:.3f}')

        best_acc = 0.0
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Finished Training')




if __name__ == '__main__':
    main()
