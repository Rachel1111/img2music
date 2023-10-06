import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from dataProcess.process import DataProcess, GetFeatures

class TensorDataProcess():

    def __init__(self, img_dir, txt_dir):
        # get image and text path
        self.img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        self.txt_paths = [os.path.join(txt_dir, filename) for filename in os.listdir(txt_dir)]

        # create Dataset if the dataset is too large, decrease
        max_length = len(self.img_paths) // 10
        self.dataset = DataProcess(self.img_paths[:max_length], self.txt_paths[:max_length])
        # create feature extracting
        self.getfeatures = GetFeatures()

    def getTensorFeatures(self):
        image_features = []
        txt_features = []

        # Iterate through the dataset for processing
        for image, txt in self.dataset:
            image_feature, txt_feature, similarity = self.getfeatures.imageAndText(image, txt)
            if similarity > 0.2:
                image_features.append(image_feature)
                txt_features.append(txt_feature)
                # print(similarity)
        image_features_tensor = torch.stack(image_features)
        txt_features_tensor = torch.stack(txt_features)
        # TensorDataset accept more than one example
        dataset = TensorDataset(image_features_tensor, txt_features_tensor)

        return dataset
#
# # 准备数据
# img_dir = r'D:\ study\master project\img2textFeatures\data\image'
# txt_dir = r'D:\study\master project\img2textFeatures\data\text'
# batch_size = 32
# learning_rate = 0.001
# num_epochs = 100
#
# getdataset = TensorDataProcess(img_dir, txt_dir)
# # 构建 Dataset 和 DataLoader
# dataset = getdataset.getTensorFeatures()