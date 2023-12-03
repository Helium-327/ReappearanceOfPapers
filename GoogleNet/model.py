import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================ #
"""LeNet模型"""


# ================ #
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()  # 输出矩阵的尺寸
        # self.conv = nn.Conv2d(3, 3, 1)
        self.conv1 = nn.Conv2d(3, 16, 5)  # (32 - 5 + 0)/1 + 1 = 28
        self.pool1 = nn.MaxPool2d(2, 2)  # 14
        self.conv2 = nn.Conv2d(16, 32, 5)  # (14 - 5 + 0)/1 + 1 = 10
        self.pool2 = nn.MaxPool2d(2, 2)  # 5
        self.fc1 = nn.Linear(32 * 5 * 5, 200)  # 120
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.conv(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


# ============= #
"""AlexNet模型"""


# ============= #

class AlexNet(nn.Module):
    # 初始化函数：模型开始时自动运行
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        """
            # 特征提取模块
            特征提取模块主要包含三种结构：
            1. 卷积       nn.Conv2d()   <---- (in_channels=?,out_channels=?,kernel_size=?,stride=?,padding=?)
            2. 激活函数    nn.ReLU()     <---- (inplace=?)
            3. 最大池化    nn.MaxPool2d()<---- (kernel_size=?, stride=?) 
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, 2),  # output = (224 - 11 + 2*2)/4 + 1 = 55  [48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # output = 55 / 2 = 27      [48, 27, 27]
            nn.Conv2d(48, 128, 5, padding=2),  # output = (27 - 5 + 2*2) / 1 + 1 = 27 [128, 27. 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # output = 27 / 2 = 13      [128, 13, 13]
            nn.Conv2d(128, 192, 3, padding=1),  # output = (13 - 3 + 2*1) / 1 + 1 = 13 [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),  # output = (13 - 3 + 2*1) / 1 + 1 = 13 [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, 3, padding=1),  # output = (13 - 3 + 2*1) / 1 + 1 = 13 [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # output = 13 / 2 =6 [128, 6, 6]
        )
        # self.features = nn.Sequential(  # Sequential容器，将一些列的操作打包成一个新的模块
        #     nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
        #     nn.ReLU(inplace=True),  # 设置inplace参数可以在内存中载入更大的模型
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
        #     nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
        #     nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        # )
        """
            # 分类器模块
            分类其模块主要包含三种结构：
            1. 随机失活  nn.Dropout()   <---- (p=?)
            2. 全连接层  nn.Linear()    <---- (input=?, output=?)
            3. 激活函数  nn.ReLU()      <---- (inplace=?)
        """
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    #   前向传播
    def forward(self, x):
        x = self.features(x)  # 对图像进行特征提取
        x = torch.flatten(x, start_dim=1)  # 按第一个维度（C）进行展平
        x = self.classifier(x)  # 将图像传入分类器
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 通过遍历modules属性，可以遍历Sequential中的所有网络结构
            """
                1.  如果遍历的层结构是nn.Conv2d（卷积层），则使用nn.init.kaiming_normal_()对权重进行初始化。
                    mode='fan_out'表示在卷积核的输出通道上进行归一化，nonlinearity='relu'表示使用ReLU激活函数。
                2.  如果遍历的层结构是nn.Linear（全连接层），则使用nn.init.normal_()对权重进行初始化。
                    0表示均值为0，0.01表示方差为0.01。nn.init.constant_()对偏置进行初始化，设置为0。
            """
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


# ============= #
"""VGGNet模型"""
# ============= #
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

model_name = 'vgg16'


class VGGNet(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGGNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # xavier_uniform()函数使得权重矩阵在每一层都有类似的方差。
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant()函数用于设置一个常数值。在这个例子中，偏置值被设置为0。
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:  # v == 卷积核个数 == 输出的通道数（Channel）
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 列表可以加乘，不能减除
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg(model_number=model_name, **kwargs):
    with open('vgg_cfgs.json', 'r') as f:
        cfgs = json.load(f)
    assert model_number in cfgs, f"Warning: model number {model_number} not in cfgs dict! "
    cfg = cfgs[model_number]

    model = VGGNet(make_features(cfg), **kwargs)
    return model


# ================ #
"""GoogleNet模型"""


# ================ #
class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)  # output = (224 - 7 + 2*3)/2 + 1 = 112
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)           # output = 112 /2 = 56

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)  # output = (56 - 1 + 0)/1 + 1 = 56
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)  # output = (56 - 3 + 2*1)/1 + 1 = 56
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # output = 56 / 2 = 28

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)  # output = (256, 28, 28)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)  # output = (480, 28. 28)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # output = (480, 14, 14)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)  # output = (512, 14, 14)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)  # output = (512, 14, 14)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)  # output = (512, 14, 14)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)  # output = (528, 14, 14)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)  # output = (832, 14, 14)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)       # 14 / 2 = 7

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)   # output = (832, 7, 7)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)   # output = (1024, 7, 7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))        # 自适应平均池化下采样
        self.dropout = nn.Dropout(p=0.4)                                #
        self.fc = nn.Linear(1024, num_classes)                          # 全连接层，全连接层节点数 = 1024

        # 条件属性
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 前向传播
        x = self.conv1(x)               # output = (224 - 7 + 2*3)/2 + 1 = 112    ( 64, 112, 112)
        x = self.maxpool1(x)            # output = 112 / 2 = 56                   ( 64, 56, 56)
        x = self.conv2(x)               # output = (56 - 1 + 0)/1+1 = 56          ( 64, 56, 56)
        x = self.conv3(x)               # output = (56 - 3 + 2*1)/1+1 = 56        ( 192, 56, 56)
        x = self.maxpool2(x)            # output = 56 / 2 = 28                    ( 192, 28, 28)

        x = self.inception3a(x)         # output = (56 - 1 + 0)/1+1 = 56          ( 64, 56, 56)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Inception(nn.Module):
    """ Inception结构的搭建"""
    # Inception参数：
    '''
        1. in_channels = ?  <--- 输入通道的数量
        2. ch1x1 = ?        <--- 1x1卷积核的数量
        3. ch3x3red = ?     <--- 图中红色3x3的卷积核的数量
        4. ch3x3 = ?        <--- 图中蓝色3x3的卷积核的数量
        5. ch5x5red = ?     <--- 图中红色5x5的卷积核的数量
        6. ch5x5 = ?        <--- 图中蓝色5x5的卷积核的数量
        7. pool_proj = ?    <--- ？
    '''
    # Inception的通道数
    '''
        output_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        output_size 不变
    '''

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 第一个分支，包含一个卷积层，                   C =  ch1x1
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        # 第二个分支，包含两个卷积层                    C = ch3x3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        # 第三个分支，包含两个卷积                      C = ch5x5

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
        # 第四个分支包含一个最大池化层和一个卷积层,       C = pool_proj

    def forward(self, x):
        # Inception的前向传播
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # cat四个输出，cat的参数需要一个tensor


class InceptionAux(nn.Module):
    # 定义辅助分类器
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)  # 平均池化层
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # 卷积层

        self.fc1 = nn.Linear(2048, 1024)  # 全连接层
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)

        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)

        x = F.dropout(x, 0.5, training=self.training)
        # x = F.relu(self.fc2(x), inplace=True)
        x = self.fc2(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
