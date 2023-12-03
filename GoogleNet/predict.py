import os
import json
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import LeNet, AlexNet, vgg, GoogleNet

num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = GoogleNet


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else cpu)

    """图片预处理"""
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """准备数据"""
    ima_path = "../data/1.jpg"
    assert os.path.exists(ima_path), f"file: '{ima_path}' does not exist."
    img = Image.open(ima_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)           # 插入一个新维度
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist"

    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    """定义预测模型"""
    model = model_name(num_classes=num_classes).to(device)
    # model = vgg(model_number=model_name, num_classes=num_classes).to(device)  # VGGNet
    weights_path = f'../models/{model_name.__name__}.pth'
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model((img).to(device))).cpu()
        # predict = torch.max(outputs, dim=1)[1].numpy()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print_res = f"class: {class_indict[str(predict_cla)]}  prob: {predict[predict_cla].numpy():.3}"
    plt.title(print_res)
    for i in range(len(predict)):
        print(f"class : {class_indict[str(i)]} prob: {predict[i].numpy():.3f}")
    plt.show()

if __name__ == '__main__':
    main()



