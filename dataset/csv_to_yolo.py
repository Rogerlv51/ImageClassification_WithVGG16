import pandas as pd
import os

data_dir = r"./dataset/football/test_label.csv"

label = pd.read_csv(data_dir, header=None)

image_name = label.iloc[:, 0]
image_label = label.iloc[:, 1]
for i, name in enumerate(image_name):
    with open(os.path.join(r"C:\\Users\\94960\\Desktop\\图像分类Demo\\dataset\\football\\test_label", name.split('.')[0] + ".txt"), 'w') as fp:
        fp.write(image_label[i].split('[')[1].split(']')[0])
