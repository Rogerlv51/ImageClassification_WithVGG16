import sys
import os
from matplotlib import pyplot as plt
current_dir = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(current_dir)[0]
sys.path.append(rootpath)
# 上面三行是为了防止vscode里面无法导入自定义的包，还有要注意不同编译器对图片等路径的规定不同
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Augmentation.data_augment import DataAugment
import numpy as np
import torch
import cv2


class ReadYOLO(Dataset):

    def __init__(self, phase="train", trans=None, device=None):
        """
        :param phase: 数据类型
            "train": 训练数据
            "valid": 验证数据
            "test": 测试数据
        :param trans: 是否进行图像增强
        """
        super(ReadYOLO, self).__init__()
        self.device = device
        self.type = type
        self.trans = trans
        self.phase = phase
        # if phase != "test":    # 有些数据集测试数据可能没有label
        #     self.labels = os.listdir(os.path.join('./dataset/', phase, 'label'))
        self.labels = os.listdir(os.path.join('.\\dataset\\football', self.phase + '_label'))  # 标签文件的根目录的所有的txt文件的名字
        self.imgs = os.listdir(os.path.join('.\\dataset\\football', self.phase + '_image'))  # 带.jpg后缀的图片名称
        self.imgnames = list(map(lambda x: x.split('.')[0], self.imgs))  # 不带格式.jpg后缀的图片名称

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        list_target = []    # 把一个txt文件里面所有的行变成列表储存在这个大列表里，用于后面拼接成array
        # 拿出txt文件对应的jpg文件，如果labels里面的名称和imgnames里面相等的地方返回True，把为True的index提出出来就是对应的文件
        # 这一步取出来的是带jpg后缀的名称
        img = self.imgs[list(map(lambda x: x == self.labels[index].split('.')[0], self.imgnames)).index(True)]
        img_dir = os.path.join('.\\dataset\\football', self.phase + '_image', img)  # 读取图片在文件夹中的真实地址
        with open(os.path.join('.\\dataset\\football', self.phase + '_label', self.labels[index]), 'r') as fp:
            for line in fp.readlines():
                if len(line.strip('\n')) > 0:
                    nums = line.strip().split(' ')
                    li = [*map(lambda x: float(x), nums)]   # == list(map(lambda x: float(x), nums))
                    list_target.append(li)   # 把txt文件中的分类标签取出来
        if len(list_target) == 0:
            array_target = np.array([])
        else:
            array_target = np.concatenate(list_target, axis=0).reshape(len(list_target), -1)
        picture = cv2.imread(img_dir)   # array = [w, h, 3]，图片读进来
        if self.trans:
            # 这里传参是根据我们自己定义的数据增强函数：detect_resize(self, img, label, size)
            picture, array_target = self.trans(picture, array_target, (224, 224))   # picture的shape:[3,w,h]
            return picture.unsqueeze(0).to(self.device), torch.from_numpy(array_target).to(self.device)
        else:
            return picture, array_target

if __name__ == '__main__':
    
    def colle(batch):
        # 假设一个batch_size=2:那么batch的shape就是((picture1_tensor, picture1_label), (picture2_tensor, picture2_label))
        imgs, targets = list(zip(*batch))  # 这里通过解压batch把多张图片的picture_tensor放在一起，picture_label放在一起
        imgs = torch.cat(imgs, dim=0)
        targets = torch.cat(targets, dim=0)
        return imgs, targets

    data_augment = DataAugment()
    dataset = ReadYOLO(trans=data_augment)
    data = iter(DataLoader(dataset, batch_size=4, drop_last=False, collate_fn=colle))
    imgs, labels = next(data)
    print(imgs) # [channels, w, h]
    # plt或者opencv都要求通道数在最后一维上，所以转置一下
    cv2.imshow("img", imgs[0].permute(1,2,0).numpy())
    cv2.waitKey()